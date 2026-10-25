#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
SFXD = ROOT / "analysis/scalefix_2026-10-06"
OUT = Path(__file__).resolve().parent
TAG = "2026-10-24"
LIMIT = 0.75
PAPER = {"S1": "S1", "S3": "X1", "S2": "S2", "S4": "X2"}
PRIMARY = {"S1": "similarity_height_metalctl", "S2": "affine_height_metalctl", "S3": "similarity_height",
           "S4": "similarity_height_metalctl"}
ALL6 = [("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S2", "S3"), ("S2", "S4"), ("S3", "S4")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def engine(case, post, rule, metalctl):
    sys.path.insert(0, str(SFXD))
    import sfx_measure as SM
    SM.TOL_GATE = LIMIT
    if metalctl:
        sys.argv = [str(SFXD / "sfx_metalctl.py"), case, post, rule]
        runpy.run_path(str(SFXD / "sfx_metalctl.py"), run_name="__main__")
    else:
        sys.argv = [SM.__file__, case, post, rule]
        SM.main()


def speccurve(case):
    spec = importlib.util.spec_from_file_location("specification_curve", ROOT / "analysis/specification_curve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.TOL = LIMIT
    m.run(case, "A")


def window_check(M):
    out = {}
    for R in ("A", "B"):
        rows = {}
        for nm, w in M["readers"][R]["controls"]["windows"].items():
            if not w.get("valid"):
                rows[nm] = {"used_at_0.90": False}
                continue
            sh, di, bo = w.get("rigid_shift_mm"), w.get("dilation_mm"), w.get("buccal_median_mm")
            rows[nm] = {"used_at_0.90": True, "buccal_median_mm": bo, "rigid_shift_mm": sh, "dilation_mm": di,
                        "palatal_witness_fails_at_0.75": bool(sh is not None and (abs(sh) > LIMIT or abs(di) > LIMIT)),
                        "buccal_reading_exceeds_0.75": bool(abs(bo) > LIMIT)}
        out[R] = rows
    return out


def env(vals):
    return [min(vals), max(vals)] if vals else None


def rel(e1, e2):
    lo, hi = max(e1[0], e2[0]), min(e1[1], e2[1])
    return round(lo - hi, 1)


def collect(r075):
    r075 = Path(r075)
    outp = OUT / f"CONTROL_LIMIT_075_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    sys.path.insert(0, str(SFXD))
    import numpy as np
    from sfx_common import V
    inputs = {}
    res = {"windows_failing_at_0.75": {}, "revised": {}, "original": {}}
    for an, tagf in (("revised", lambda c: PRIMARY[c]), ("original", lambda c: "frozen_old_final2")):
        res["windows_failing_at_0.75"][an] = {}
        for c in ("S1", "S3", "S2", "S4"):
            p = SFXD / c / f"measure_{tagf(c)}.json"; inputs[str(p.relative_to(ROOT))] = sha(p)
            res["windows_failing_at_0.75"][an][PAPER[c]] = window_check(json.loads(p.read_text()))
    A90 = json.loads((SFXD / "FINAL_scalefix_metalctl_2026-10-06/SCALEFIX_METALCTL_RESULTS_2026-10-06.json").read_text())
    p75 = r075 / "analysis/scalefix_2026-10-06/FINAL_scalefix_metalctl_2026-10-06/SCALEFIX_METALCTL_RESULTS_2026-10-06.json"
    A75 = json.loads(p75.read_text()); inputs["root_075:" + str(p75.relative_to(r075))] = sha(p75)
    rv = {}
    for c in ("S1", "S3", "S2", "S4"):
        a, b = A90["cases"][c], A75["cases"][c]
        rv[PAPER[c]] = {"primary_0.90": {R: a["readers"][R]["primary"]["net_mm3"] for R in "AB"},
                        "primary_0.75": {R: b["readers"][R]["primary"]["net_mm3"] for R in "AB"},
                        "envelope_common_family_0.90": [a["envelopes"]["common_family"]["case"]["min_mm3"], a["envelopes"]["common_family"]["case"]["max_mm3"]],
                        "envelope_common_family_0.75": [b["envelopes"]["common_family"]["case"]["min_mm3"], b["envelopes"]["common_family"]["case"]["max_mm3"]],
                        "envelope_all_exact_0.90": [a["envelopes"]["all_exact"]["case"]["min_mm3"], a["envelopes"]["all_exact"]["case"]["max_mm3"]],
                        "envelope_all_exact_0.75": [b["envelopes"]["all_exact"]["case"]["min_mm3"], b["envelopes"]["all_exact"]["case"]["max_mm3"]],
                        "n_members_0.90": len(a["members"]["A"]) + len(a["members"]["B"]),
                        "n_members_0.75": len(b["members"]["A"]) + len(b["members"]["B"])}
        rv[PAPER[c]]["unchanged"] = all(rv[PAPER[c]][f"{k}_0.90"] == rv[PAPER[c]][f"{k}_0.75"]
                                        for k in ("primary", "envelope_common_family", "envelope_all_exact", "n_members"))
    rv["pairs"] = {k: {"reading_0.90": A90["pairs"]["main"][k]["reading"], "reading_0.75": A75["pairs"]["main"][k]["reading"],
                       "gaps_0.90": {v: g["signed_gap_mm3"] for v, g in A90["pairs"]["main"][k]["views"].items()},
                       "gaps_0.75": {v: g["signed_gap_mm3"] for v, g in A75["pairs"]["main"][k]["views"].items()}}
                   for k in A90["pairs"]["main"]}
    res["revised"] = rv
    og = {}
    for c in ("S1", "S3", "S2", "S4"):
        p = SFXD / c / "measure_frozen_old_final2.json"; M = json.loads(p.read_text())
        q = r075 / f"analysis/scalefix_2026-10-06/{c}/measure_frozen_old.json"; M75 = json.loads(q.read_text())
        inputs["root_075:" + str(q.relative_to(r075))] = sha(q)
        Z = np.load(ROOT / M["fields_npz"])
        rows = {}
        for R in "AB":
            rr = M["readers"][R]; wc = window_check(M)[R]
            keep = [nm for nm, w in wc.items() if w["used_at_0.90"] and not w["palatal_witness_fails_at_0.75"]]
            refused = any(wc[nm]["buccal_reading_exceeds_0.75"] for nm in keep)
            kept = [nm for nm in keep if not wc[nm]["buccal_reading_exceeds_0.75"]]
            Hpre, Hpost, Hc = Z[f"{R}_primary_10x10_Hpre"], Z[f"{R}_primary_10x10_Hpost"], Z[f"{R}_primary_10x10_d1.4_Hc"]
            xe, ye = Z[f"{R}_primary_10x10_xe"], Z[f"{R}_primary_10x10_ye"]
            imp, pin = rr["as_executed"]["implant_in_envelope_mm3"], rr["primary"]["pin_metal_mm3"]
            if kept:
                bg = float(np.mean([wc[nm]["buccal_median_mm"] for nm in kept]))
                mw = lambda H: V.measure_window(None, None, None, None, 0.2, background=bg, fields=(Hpre, H, xe, ye))["gain_mm3"]
                drop = {"windows_used": kept, "background_mm": round(bg, 4), "as_executed_net_mm3": round(mw(Hpost) - imp, 1),
                        "pin_corrected_net_mm3": round(mw(Hc) - imp - pin, 1)}
            else:
                drop = None
            rows[R] = {"original_net_0.90": rr["as_executed"]["net_mm3"], "pin_corrected_net_0.90": rr["primary"]["net_mm3"],
                       "run_refused_at_0.75": refused, "with_failing_window_left_out_0.75": drop,
                       "engine_rerun_0.75": {"as_executed_net_mm3": M75["readers"][R]["as_executed"]["net_mm3"],
                                             "pin_corrected_net_mm3": M75["readers"][R]["primary"]["net_mm3"],
                                             "aborts": M75["readers"][R]["controls"]["abort_conditions"],
                                             "n_admissible_specs": sum(1 for s in M75["readers"][R]["specs"] if s.get("admissible")),
                                             "n_admissible_specs_0.90": sum(1 for s in rr["specs"] if s.get("admissible"))}}
        og[PAPER[c]] = rows
    envs = {}
    for c in ("S1", "S3", "S2", "S4"):
        s90 = json.loads((ROOT / f"analysis/{c}/specification_curve.json").read_text())
        s75p = r075 / f"analysis/{c}/specification_curve.json"; s75 = json.loads(s75p.read_text())
        inputs["root_075:" + str(s75p.relative_to(r075))] = sha(s75p)
        op = json.loads((ROOT / f"analysis/{c}/outline_compare.json").read_text())["calibrated_windowed_mm3"]
        fa = json.loads((ROOT / f"analysis/{c}/results.json").read_text())["volume"]["PRIMARY_net_bone_gain_mm3"]
        fb = json.loads((ROOT / f"analysis/{c}/readerB/results.json").read_text())["volume"]["PRIMARY_net_bone_gain_mm3"]
        sp = lambda S: [s["net_mm3"] for s in S["specifications"] if s["roi_w"] == 10 and s["roi_d"] == 10]
        b75 = og[PAPER[c]]["B"]
        b_alt = None if not b75["run_refused_at_0.75"] else (b75["with_failing_window_left_out_0.75"] or {}).get("as_executed_net_mm3")
        e90 = env([fa] + sp(s90) + [op] + [fb])
        e75_refused = env([fa] + sp(s75) + [op] + ([] if b75["run_refused_at_0.75"] else [fb]))
        e75_left_out = env([fa] + sp(s75) + [op] + [b_alt if b_alt is not None else fb])
        envs[c] = {"0.90": e90, "0.75_refused_runs_dropped": e75_refused, "0.75_failing_window_left_out": e75_left_out,
                   "n_specs_10x10_0.90": len(sp(s90)), "n_specs_10x10_0.75": len(sp(s75))}
    opairs = {f"{PAPER[a]}_{PAPER[b]}": {k: rel(envs[a][k], envs[b][k]) for k in ("0.90", "0.75_refused_runs_dropped",
                                                                                     "0.75_failing_window_left_out")}
              for a, b in ALL6}
    res["original"] = {"values": og, "envelopes": {PAPER[c]: v for c, v in envs.items()}, "pair_signed_gaps_mm3": opairs}
    out = {"_what": "",
           "limit_mm": LIMIT, **res, "inputs_sha256": inputs, "script_sha256": sha(Path(__file__).resolve()),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(outp, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(res["windows_failing_at_0.75"], indent=0)[:3000])
    print({k: v["unchanged"] for k, v in rv.items() if k != "pairs"})
    print(json.dumps(og, indent=0)[:3000]); print(json.dumps(envs, indent=0)); print(json.dumps(opairs, indent=0))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 4 and a[0] == "--engine":
        engine(a[1], a[2], a[3], "--metalctl" in a)
    elif len(a) == 2 and a[0] == "--speccurve":
        speccurve(a[1])
    elif len(a) == 2 and a[0] == "--collect":
        collect(a[1])
    else:
        raise SystemExit("usage: --engine CASE POST RULE [--metalctl] | --speccurve CASE | --collect ROOT_075")

