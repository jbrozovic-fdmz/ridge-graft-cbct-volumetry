#!/usr/bin/env python3
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
SFXD = ROOT / "analysis/scalefix_2026-10-06"
sys.path.insert(0, str(SFXD))
import sfx_measure as SM
from sfx_common import geom, V, READERS, post_path, sha256_file, write_new, rss_gib

OUT = Path(__file__).resolve().parent
TAG = "2026-10-24"
LEVELS = tuple(float(v) for v in range(300, 801, 50))
PRIMARY = {"S1": "similarity_height_metalctl", "S2": "affine_height_metalctl", "S3": "similarity_height",
           "S4": "similarity_height_metalctl"}
STORED_PRIMARY = {"S1": (99.7, 99.5), "S3": (137.0, 138.3), "S2": (95.4, 83.6), "S4": (267.1, 271.7)}
PAPER = {"S1": "S1", "S3": "X1", "S2": "S2", "S4": "X2"}
DEPTH_JSON = ROOT / "FINAL_ARCH_20260907/containment_NULLCORRECTED.json"


def controls_at(c, R, P, Q, MR):
    z0, z1 = c.bands[R]; KEEP = c.KEEP[R]
    meds, wins, flags = [], {}, []
    for nm, w in MR["controls"]["windows"].items():
        if not w.get("valid"):
            wins[nm] = {"used": False}
            continue
        b = V.measure_window(P, Q, w["X_mm"], w["Y_mm"], SM.GRID, z0, z1, keep_box=KEEP)
        rec = {"used": True, "X_mm": w["X_mm"], "Y_mm": w["Y_mm"], "cells": b.get("cells", 0)}
        if b.get("cells", 0) == 0:
            flags.append(f"{nm}: no cells")
            wins[nm] = rec
            continue
        bo = b["median_thickness_mm"]
        rec["buccal_median_mm"] = bo
        if b["cells"] < 200:
            flags.append(f"{nm}: only {b['cells']} cells")
        lw = V.measure_window(P, Q, w["X_mm"], w["Y_mm"], SM.GRID, z0, z1, mode="bottom_mean", keep_box=KEEP)
        if lw.get("cells", 0) >= 200:
            lo = -lw["median_thickness_mm"]
            sh, di = (bo - lo) / 2.0, (bo + lo) / 2.0
            rec.update(lingual_outward_mm=round(lo, 3), rigid_shift_mm=round(sh, 3), dilation_mm=round(di, 3))
            if not (abs(sh) <= SM.TOL_GATE and abs(di) <= SM.TOL_GATE):
                flags.append(f"{nm}: palatal-witness test would fail")
        if abs(bo) > SM.TOL_GATE:
            flags.append(f"{nm}: buccal median {bo:+.3f} mm exceeds {SM.TOL_GATE}")
        meds.append(bo)
        wins[nm] = rec
    if len(meds) > 1 and (max(meds) - min(meds)) > 2 * SM.TOL_GATE:
        flags.append(f"controls disagree by {max(meds) - min(meds):.3f} mm")
    bg = float(np.mean(meds)) if meds else None
    return bg, wins, flags


def one(c, R, P, Q, MR):
    roi = c.res[R]["roi"]; z0, z1 = c.bands[R]; KEEP = c.KEEP[R]
    m0 = V.measure_window(P, Q, roi["X_mm"], roi["Y_mm"], SM.GRID, z0, z1, background=0.0, keep_box=KEEP)
    Hpre, Hpost, xe, ye = m0["_Hpre"], m0["_Hpost"], m0["_x_edges"], m0["_y_edges"]
    bg, wins, flags = controls_at(c, R, P, Q, MR)
    if bg is None:
        return {"background_mm": None, "controls": wins, "qc_flags": flags + ["no control measured"]}
    main = V.measure_window(None, None, None, None, SM.GRID, background=bg, fields=(Hpre, Hpost, xe, ye))
    bg_p = MR["controls"]["background_unrounded"]
    gross_pbg = V.measure_window(None, None, None, None, SM.GRID, background=bg_p, fields=(Hpre, Hpost, xe, ye))
    imp_t, _ = SM.ti_term(c, R, c.implant_idx, Hpre, Hpost, xe, ye)
    prom = {L: bool(MR["prominence"][str(L)]["cap_applied"]) for L in c.elig}
    Hc, F_all, Fcap, ties = SM.d7e_surface(c, R, xe, ye, Hpre, Hpost, prom, SM.D_PRIMARY)
    m = V.measure_window(None, None, None, None, SM.GRID, background=bg, fields=(Hpre, Hc, xe, ye))
    pin_t, _ = SM.ti_term(c, R, c.pin_idx, Hpre, Hc, xe, ye)
    net = round(m["gain_mm3"] - imp_t - pin_t, 1)
    return {"net_mm3": net, "gross_mm3": main["gain_mm3"], "gross_after_cap_mm3": m["gain_mm3"],
            "implant_mm3": imp_t, "pin_metal_mm3": pin_t,
            "cap_term_mm3": round(SM.gain_u(Hpre, Hc, bg) - SM.gain_u(Hpre, Hpost, bg), 4),
            "background_mm": round(bg, 4), "background_unrounded": bg, "coverage": m["coverage"],
            "area_mm2": m["area_mm2"], "gross_at_primary_background_mm3": gross_pbg["gain_mm3"],
            "capped_cells": int(Fcap.sum()), "pin_footprint_cells": int(F_all.sum()),
            "controls": wins, "qc_flags": flags}


def run(case):
    tag = PRIMARY[case]
    outp = OUT / f"THRESHOLD_SWEEP_NET_{case}_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    t0 = time.time()
    mj = SFXD / case / f"measure_{tag}.json"
    M = json.loads(mj.read_text())
    post = M["post"]
    c = SM.setup(case, post, str(DEPTH_JSON))
    c.surfaces = None; c.P = None; c.Q = None
    pf = geom.bone_field(c.pre_c); qf = geom.bone_field(c.post_c)
    rows = {R: {} for R in READERS}
    for lv in LEVELS:
        Vp, Fp = geom.iso_surface(pf, c.vx, level=lv); Vp = Vp + c.origin_mm
        Vq, Fq = geom.iso_surface(qf, c.vx, level=lv); Vq = Vq + c.origin_mm
        for R in READERS:
            P = geom.surface_points_local(Vp, Fp, c.frames[R], c.KEEP[R])
            Q = geom.surface_points_local(Vq, Fq, c.frames[R], c.KEEP[R])
            rows[R][f"{lv:.0f}"] = one(c, R, P, Q, M["readers"][R])
            print(case, R, f"{lv:.0f}", {k: rows[R][f'{lv:.0f}'].get(k) for k in
                                          ("net_mm3", "gross_mm3", "implant_mm3", "pin_metal_mm3", "background_mm")},
                  rows[R][f"{lv:.0f}"]["qc_flags"], flush=True)
            del P, Q
        del Vp, Fp, Vq, Fq
    del pf, qf
    gate = {}
    for i, R in enumerate(READERS):
        MR = M["readers"][R]; g = rows[R]["450"]
        chk = {"net": [g["net_mm3"], MR["primary"]["net_mm3"], STORED_PRIMARY[case][i]],
               "background_unrounded": [g["background_unrounded"], MR["controls"]["background_unrounded"]],
               "gross": [g["gross_mm3"], MR["as_executed"]["gross_mm3"]],
               "implant": [g["implant_mm3"], MR["as_executed"]["implant_in_envelope_mm3"]],
               "pin_metal": [g["pin_metal_mm3"], MR["primary"]["pin_metal_mm3"]],
               "coverage": [g["coverage"], MR["primary"]["coverage"]]}
        if R == "A" and "threshold_sweep_gross_mm3" in MR:
            for k, v in MR["threshold_sweep_gross_mm3"].items():
                chk[f"stored_gross_sweep_{k}"] = [rows[R][k]["gross_at_primary_background_mm3"], v]
        gate[R] = {k: {"new": v[0], "stored": v[1:], "equal": all(v[0] == x for x in v[1:])} for k, v in chk.items()}
    passed = all(x["equal"] for R in READERS for x in gate[R].values())
    out = {"_what": "",
           "case": case, "paper_code": PAPER[case], "primary_tag": tag, "levels": [f"{v:.0f}" for v in LEVELS],
           "regression_gate_450": gate, "gate_passed": passed, "readers": rows,
           "inputs": {str(mj.relative_to(ROOT)): sha256_file(mj),
                      str(post_path(case, post).relative_to(ROOT)): sha256_file(post_path(case, post)),
                      f"analysis/{case}/pre_cal.nii.gz": sha256_file(ROOT / f"analysis/{case}/pre_cal.nii.gz")},
           "script": str(Path(__file__).resolve().relative_to(ROOT)),
           "script_sha256": sha256_file(Path(__file__).resolve()),
           "engine_sha256": sha256_file(Path(SM.__file__).resolve()),
           "peak_rss_GiB": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    print(case, "gate", "PASSED" if passed else "FAILED", json.dumps(gate), flush=True)
    if not passed:
        raise SystemExit(f"{case}: regression gate failed")


def intervals(levels, mask):
    out, start = [], None
    for lv, m in zip(levels, mask):
        if m and start is None:
            start = lv
        if not m and start is not None:
            out.append([start, prev]); start = None
        prev = lv
    if start is not None:
        out.append([start, prev])
    return out


def collect():
    outp = OUT / f"THRESHOLD_SWEEP_NET_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    per = {c: json.loads((OUT / f"THRESHOLD_SWEEP_NET_{c}_{TAG}.json").read_text()) for c in PRIMARY}
    assert all(per[c]["gate_passed"] for c in per)
    lv = per["S1"]["levels"]
    pairs = {}
    for R in READERS:
        n = {c: [per[c]["readers"][R][k]["net_mm3"] for k in lv] for c in per}
        d_imp = [round(n["S3"][i] - n["S1"][i], 1) for i in range(len(lv))]
        d_noimp = [round(n["S4"][i] - n["S2"][i], 1) for i in range(len(lv))]
        s1_gt = [n["S1"][i] > n["S3"][i] for i in range(len(lv))]
        pairs[R] = {"X1_minus_S1_mm3": dict(zip(lv, d_imp)), "X2_minus_S2_mm3": dict(zip(lv, d_noimp)),
                    "order_implant_pair": {k: ("X1 > S1" if d > 0 else "S1 > X1" if d < 0 else "equal") for k, d in zip(lv, d_imp)},
                    "order_no_implant_pair": {k: ("X2 > S2" if d > 0 else "S2 > X2" if d < 0 else "equal") for k, d in zip(lv, d_noimp)},
                    "levels_S1_gt_X1": intervals([int(k) for k in lv], s1_gt),
                    "X2_minus_S2_min_mm3": min(d_noimp), "X2_minus_S2_max_mm3": max(d_noimp),
                    "X1_minus_S1_min_mm3": min(d_imp), "X1_minus_S1_max_mm3": max(d_imp)}
    flags = {PAPER[c]: {R: {k: per[c]["readers"][R][k]["qc_flags"] for k in lv if per[c]["readers"][R][k]["qc_flags"]}
                        for R in READERS} for c in per}
    table = {PAPER[c]: {R: {k: {f: per[c]["readers"][R][k].get(f) for f in
                                ("net_mm3", "gross_mm3", "gross_at_primary_background_mm3", "implant_mm3",
                                 "pin_metal_mm3", "cap_term_mm3", "background_mm", "coverage")} for k in lv}
                        for R in READERS} for c in per}
    out = {"_what": "",
           "case_codes": PAPER, "levels": lv, "table": table, "pairs": pairs, "qc_flags": flags,
           "per_case_files": {PAPER[c]: {"file": f"THRESHOLD_SWEEP_NET_{c}_{TAG}.json",
                                         "sha256": sha256_file(OUT / f"THRESHOLD_SWEEP_NET_{c}_{TAG}.json"),
                                         "peak_rss_GiB": per[c]["peak_rss_GiB"], "gate_passed": per[c]["gate_passed"]}
                              for c in per},
           "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256_file(Path(__file__).resolve()),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    with open(OUT / f"THRESHOLD_SWEEP_NET_{TAG}.csv", "x", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "assessor", "threshold", "net_mm3", "gross_mm3", "gross_at_primary_background_mm3",
                    "implant_mm3", "pin_metal_mm3", "cap_term_mm3", "background_mm", "coverage", "qc_flags"])
        for c in ("S1", "S3", "S2", "S4"):
            for R in READERS:
                for k in lv:
                    r_ = per[c]["readers"][R][k]
                    w.writerow([PAPER[c], R, k, r_["net_mm3"], r_["gross_mm3"], r_["gross_at_primary_background_mm3"],
                                r_["implant_mm3"], r_["pin_metal_mm3"], r_["cap_term_mm3"], r_["background_mm"],
                                r_["coverage"], "; ".join(r_["qc_flags"])])
    print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--case" and a[1] in PRIMARY:
        run(a[1])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit("usage: --case S1|S2|S3|S4 | --collect")

