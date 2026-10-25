#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "analysis"))

TAG3 = "v3_2026-09-14"
TAG = "D7e_2026-09-14"
CORE = (0.0, 0.5)
RING = (1.4, 3.0)
MIN_CELLS = 5
TOL = 1e-6
VERIFIER = ROOT / "JFB_REVISION_2026-09-13/04_validation"
CHECK = {"S2": ["64"], "S4": ["760"]}
CHECK_RINGS = ((0.0, 0.5), (1.4, 2.0), (2.0, 3.0))
OUT_FINAL = ROOT / f"FINAL_ARCH_20260907/pin_prominence_{TAG}.json"


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, obj):
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def free_avail_g():
    return float(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()[-1])


def dist_field(xe, ye, pin_xy):
    from scipy.spatial import cKDTree
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    G = np.column_stack([X.ravel(), Y.ravel()])
    d, _ = cKDTree(np.asarray(pin_xy, float)[:, :2]).query(G, k=1)
    return d.reshape(X.shape)


def band_stats(Hpost, Hpre, mask, ztop):
    fin = np.isfinite(Hpost) & np.isfinite(Hpre)
    m = mask & fin
    m_post = mask & np.isfinite(Hpost)
    h = Hpost[m] - ztop
    hp = Hpost[m_post] - ztop
    return {"cells_in_band": int(mask.sum()), "valid_cells": int(m.sum()),
            "median_Hpost_minus_Ztop_mm": r(np.median(h), 4) if len(h) else None,
            "q25_q75_mm": [r(q, 3) for q in np.quantile(h, [.25, .75])] if len(h) else None,
            "_unrounded": float(np.median(h)) if len(h) else None,
            "QC_valid_cells_Hpost_only": int(m_post.sum()),
            "QC_median_Hpost_only_mm": r(np.median(hp), 4) if len(hp) else None,
            "_unrounded_post_only": float(np.median(hp)) if len(hp) else None}


def prominence(Hpost, Hpre, D, lab, elig, ztop, vox):
    core = {L: D[L] <= CORE[1] + TOL for L in elig}
    other = np.zeros_like(core[lab])
    for L in elig:
        if L != lab:
            other |= core[L]
    cmask = core[lab] & ~other
    rmask = (D[lab] > RING[0]) & (D[lab] <= RING[1]) & ~other
    c = band_stats(Hpost, Hpre, cmask, ztop)
    g = band_stats(Hpost, Hpre, rmask, ztop)
    rec = {"core_0-0.5mm": c, "ring_1.4-3.0mm": g,
           "cells_excluded_as_other_pin_core": {"core": int((core[lab] & other).sum()),
                                                "ring": int(((D[lab] > RING[0]) & (D[lab] <= RING[1]) & other).sum())},
           "cells_within_1e-6_of_band_edges": int(sum((np.abs(D[lab] - e) < TOL).sum() for e in (0.5, 1.4, 3.0)))}
    if c["valid_cells"] < MIN_CELLS or g["valid_cells"] < MIN_CELLS:
        rec.update(P_mm=None, P_unrounded=None, decision="undetermined", cap_applied=False)
    else:
        P = c["_unrounded"] - g["_unrounded"]
        rec.update(P_mm=r(P, 3), P_unrounded=P, decision="prominent" if P > vox else "not prominent",
                   cap_applied=bool(P > vox))
    if c["QC_valid_cells_Hpost_only"] >= MIN_CELLS and g["QC_valid_cells_Hpost_only"] >= MIN_CELLS:
        Pp = c["_unrounded_post_only"] - g["_unrounded_post_only"]
        rec["QC_P_Hpost_only_mm"] = r(Pp, 3)
        rec["QC_decision_Hpost_only"] = "prominent" if Pp > vox else "not prominent"
    else:
        rec["QC_P_Hpost_only_mm"] = None
        rec["QC_decision_Hpost_only"] = "undetermined"
    for b in (c, g):
        b.pop("_unrounded"); b.pop("_unrounded_post_only")
    return rec


def run_case(case):
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from pipeline import geom
    outp = ROOT / f"analysis/{case}/pin_prominence_{TAG}.json"
    assert not outp.exists(), f"{outp} exists; not overwritten"
    rule3 = json.loads((ROOT / f"FINAL_ARCH_20260907/pin_artefact_rule_{TAG3}.json").read_text())
    elig = rule3["eligible_pins_per_case"][case]
    vox = float(rule3["voxel_mm_cap_offset_per_case"][case])
    corr = json.loads((ROOT / f"analysis/{case}/pin_artefact_corrected_{TAG3}.json").read_text())
    npz_p = ROOT / f"analysis/{case}/pin_artefact_fields_{TAG3}.npz"
    assert sha256_file(npz_p) == corr["fields_npz_sha256"], "frozen fields npz changed"
    assert corr["eligible_pins"] == elig
    avail = free_avail_g()
    assert avail >= 6, f"ABORT: only {avail} GB available"
    inv = {p["label"]: p for p in json.loads((ROOT / "FINAL_ARCH_20260907/fixation_pins_2026-09-13.json").read_text())
           ["inventory_near_site"][case]["pin_candidates"]}
    au = json.loads((ROOT / f"analysis/{case}/fixation_pins_2026-09-13.json").read_text())
    lo = np.array(au["crop_lo_vox"]); hi = np.array(au["crop_hi_vox"])
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    img = sitk.ReadImage(str(ROOT / f"cbct_blinded/{case}/post_reg.nii.gz"))
    vx = float(img.GetSpacing()[0])
    assert abs(vx - vox) < 1e-6, (vx, vox)
    post = sitk.GetArrayViewFromImage(img)[sl].astype(np.float32)
    del img
    lbl, _ = ndi.label(post > 2800.0)
    sizes = np.bincount(lbl.ravel())
    pins = {}
    for L in elig:
        cand = [k for k in np.nonzero(sizes == inv[L]["n_voxels"])[0] if k > 0]
        cand = [k for k in cand if round(float(post[lbl == k].max())) == round(inv[L]["post_grey_max"])]
        assert len(cand) == 1, (L, cand)
        pins[L] = np.argwhere(lbl == cand[0]) + lo
        assert len(pins[L]) == corr["eligible_pin_voxels"][str(L)]
    del post, lbl
    z = np.load(npz_p)
    from importlib.util import spec_from_file_location, module_from_spec
    out = {"case": case, "voxel_mm": vox, "eligible_pins": elig, "free_g_available_at_load": avail,
           "fields_npz": str(npz_p.relative_to(ROOT)), "fields_npz_sha256": corr["fields_npz_sha256"],
           "readers": {}}
    for R in ("A", "B"):
        res = json.loads((ROOT / (f"analysis/{case}/results.json" if R == "A"
                                  else f"analysis/{case}/readerB/results.json")).read_text())
        frame = geom.LocalFrame.from_dict(res["frame"])
        loc = {L: frame.to_local(pins[L].astype(float) * vx) for L in elig}
        ztop = {L: float(loc[L][:, 2].max()) for L in elig}
        windows = sorted({k.split("_", 1)[1].rsplit("_", 1)[0] for k in z.files if k.startswith(R + "_")
                          and k.endswith("_Hpost") and "_d" not in k})
        rr = {"windows_in_frozen_fields": windows, "pins": {}}
        from scipy.spatial import cKDTree
        for wn in windows:
            k = f"{R}_{wn}"
            allxy = np.vstack([loc[L][:, :2] for L in elig])
            for d in ("1.0", "1.4", "2.0"):
                Fm = dist_field(z[k + "_xe"], z[k + "_ye"], allxy) <= float(d) + TOL
                assert np.array_equal(Fm, z[f"{k}_d{d}_F"]), ("footprint not reproduced", R, wn, d)
        rr["stored_v3_union_footprint_reproduced_all_windows_d"] = True
        entry = {str(L): {wn: {dd: corr["readers"][R][wn]["variants"][dd]["per_pin"][str(L)]["footprint_cells_this_pin"]
                               for dd in ("d=1.0", "d=1.4", "d=2.0")} for wn in windows} for L in elig}
        for L in elig:
            st = corr["readers"][R]["primary_10x10"]["variants"]["d=1.4"]["per_pin"][str(L)]["titanium_max_Z_mm"]
            assert abs(round(ztop[L], 2) - st) < 1e-9, (L, ztop[L], st)
        per_w = {}
        for wn in windows:
            k = f"{R}_{wn}"
            xe, ye, Hpre, Hpost = z[k + "_xe"], z[k + "_ye"], z[k + "_Hpre"], z[k + "_Hpost"]
            D = {L: dist_field(xe, ye, loc[L][:, :2]) for L in elig}
            per_w[wn] = (Hpre, Hpost, D)
        chk = {}
        if case in CHECK:
            vj = json.loads((VERIFIER / f"pin_artefact_hostile_verify_radial_{case}_{TAG3}.json").read_text())
            Hpre, Hpost, D = per_w["primary_10x10"]
            Dst = np.stack([D[L] for L in elig])
            Lnear = np.array(elig)[np.argmin(Dst, 0)]
            fin = np.isfinite(Hpost) & np.isfinite(Hpre)
            for lab in CHECK[case]:
                L = int(lab)
                assert abs(round(ztop[L], 2) - vj[R][lab]["pin_top_Z_mm"]) < 1e-9
                prof = {tuple(p["ring"]): p for p in vj[R][lab]["radial_profile"]}
                for a, b in CHECK_RINGS:
                    m = fin & (Lnear == L) & ((D[L] > a) if a > 0 else (D[L] >= 0)) & (D[L] <= b)
                    mine = float(np.median(Hpost[m] - ztop[L]))
                    ref = prof[(a, b)]["frozen_Hpost_minus_pin_top_mm_q25_median_q75"][1]
                    assert int(m.sum()) == prof[(a, b)]["cells"], (R, lab, a, b, int(m.sum()), prof[(a, b)]["cells"])
                    assert abs(mine - ref) <= 0.01, ("verifier median not reproduced", R, lab, a, b, mine, ref)
                    chk[f"pin{lab}_ring{a}-{b}"] = {"mine_mm": r(mine, 4), "verifier_mm": ref, "cells": int(m.sum()),
                                                    "abs_diff_mm": r(abs(mine - ref), 4)}
            rr["verifier_reproduction_check_passed"] = True
            rr["verifier_reproduction_check"] = chk
        for L in elig:
            rec = {"Ztop_mm": r(ztop[L], 4), "titanium_voxels": int(len(pins[L])),
                   "footprint_cells_this_pin_v3_by_window_d": entry[str(L)],
                   "enters_windows_at_d1.4": [wn for wn in windows if entry[str(L)][wn]["d=1.4"] > 0]}
            Hpre, Hpost, D = per_w["primary_10x10"]
            rec["DECISION_primary_10x10"] = prominence(Hpost, Hpre, D, L, elig, ztop[L], vox)
            if "secondary_10x14" in per_w:
                Hpre2, Hpost2, D2 = per_w["secondary_10x14"]
                n0, n1 = Hpost.shape
                assert np.array_equal(Hpost, Hpost2[:n0, :n1], equal_nan=True)
                assert np.array_equal(Hpre, Hpre2[:n0, :n1], equal_nan=True)
                q = prominence(Hpost2, Hpre2, D2, L, elig, ztop[L], vox)
                q["_note"] = ""
                q["decision_agrees_with_primary"] = q["decision"] == rec["DECISION_primary_10x10"]["decision"]
                rec["QC_secondary_10x14"] = q
            p = rec["DECISION_primary_10x10"]
            p.pop("P_unrounded")
            if "QC_secondary_10x14" in rec:
                rec["QC_secondary_10x14"].pop("P_unrounded")
            rr["pins"][str(L)] = rec
            print(case, R, L, "Ztop", r(ztop[L], 3), "core", p["core_0-0.5mm"]["valid_cells"],
                  p["core_0-0.5mm"]["median_Hpost_minus_Ztop_mm"], "ring", p["ring_1.4-3.0mm"]["valid_cells"],
                  p["ring_1.4-3.0mm"]["median_Hpost_minus_Ztop_mm"], "P", p["P_mm"], p["decision"],
                  "| QC10x14", (rec.get("QC_secondary_10x14") or {}).get("P_mm"),
                  (rec.get("QC_secondary_10x14") or {}).get("decision"), flush=True)
        out["readers"][R] = rr
    out["peak_rss_GiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out["script"] = "analysis/pin_prominence_D7e_2026-09-14.py"
    out["script_sha256"] = sha256_file(Path(__file__).resolve())
    write_new(outp, out)
    print("wrote", outp, "peak RSS", out["peak_rss_GiB"], "GiB")


def collect():
    assert not OUT_FINAL.exists()
    rule3 = json.loads((ROOT / f"FINAL_ARCH_20260907/pin_artefact_rule_{TAG3}.json").read_text())
    per, table = {}, []
    for case in ("S1", "S2", "S3", "S4"):
        elig = rule3["eligible_pins_per_case"][case]
        if not elig:
            per[case] = {"eligible_pins": [], "note": ""}
            continue
        p = ROOT / f"analysis/{case}/pin_prominence_{TAG}.json"
        j = json.loads(p.read_text())
        per[case] = {"file": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "result": j}
        for R in ("A", "B"):
            for L, rec in j["readers"][R]["pins"].items():
                d = rec["DECISION_primary_10x10"]
                q = rec.get("QC_secondary_10x14")
                table.append({"case": case, "reader": R, "pin": int(L), "voxel_mm": j["voxel_mm"],
                              "Ztop_mm": rec["Ztop_mm"],
                              "core_median_mm": d["core_0-0.5mm"]["median_Hpost_minus_Ztop_mm"],
                              "core_valid_cells": d["core_0-0.5mm"]["valid_cells"],
                              "ring_median_mm": d["ring_1.4-3.0mm"]["median_Hpost_minus_Ztop_mm"],
                              "ring_valid_cells": d["ring_1.4-3.0mm"]["valid_cells"],
                              "P_mm": d["P_mm"], "decision": d["decision"], "cap_applied": d["cap_applied"],
                              "enters_windows_at_d1.4": rec["enters_windows_at_d1.4"],
                              "footprint_cells_v3_by_window_d": rec["footprint_cells_this_pin_v3_by_window_d"],
                              "QC_Hpost_only_P_mm": d["QC_P_Hpost_only_mm"],
                              "QC_Hpost_only_decision": d["QC_decision_Hpost_only"],
                              "QC_10x14_P_mm": q["P_mm"] if q else None,
                              "QC_10x14_decision": q["decision"] if q else None})
    fin = {"_what": "",
           "date": "",
           "definition": {"P": "",
                          "Ztop": "",
                          "exclusion": "",
                          "valid_cell": "",
                          "undetermined": "",
                          "threshold": "",
                          "field": "",
                          "d_independent": True},
           "reproduction_check": "",
           "table": table, "per_case": per,
           "corrected_values_computed": False,
           "script": "analysis/pin_prominence_D7e_2026-09-14.py",
           "script_sha256": sha256_file(Path(__file__).resolve()),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(OUT_FINAL, fin)
    for t in table:
        print(t["case"], t["reader"], t["pin"], t["core_valid_cells"], t["core_median_mm"], t["ring_valid_cells"],
              t["ring_median_mm"], "P", t["P_mm"], ">", round(t["voxel_mm"], 2), t["decision"], t["enters_windows_at_d1.4"],
              "QC", t["QC_Hpost_only_decision"], t["QC_10x14_P_mm"], t["QC_10x14_decision"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--case" and a[1] in ("S2", "S3", "S4"):
        run_case(a[1])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit(__doc__)

