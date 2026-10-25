#!/usr/bin/env python3
import json
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roi_extent_sweep as SWEEP
from pipeline import geom, landmarks, volume as V

WIDTH = 10.0
CONTAINMENT = Path("FINAL_ARCH_20260907/containment_NULLCORRECTED.json")
FROZEN = Path("FINAL_ARCH_20260907/FROZEN_RESULTS.json")
OUT = Path("FINAL_ARCH_20260907/secondary_contained_depth.json")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def contained_depth(case):
    c = json.loads(CONTAINMENT.read_text())["cases"][case]
    depth = 10.0
    if c["contiguous_from_window_edge"]:
        bands = sorted((tuple(float(x) for x in b.split("-")) for b in c["bands_above_null"]))
        for lo, hi in bands:
            if lo == depth:
                depth = hi
    return depth


def run(case, reader):
    base = json.loads(Path(f"cases/{case}.json").read_text())
    rp = (f"analysis/{case}/results.json" if reader == "A"
          else f"analysis/{case}/reader{reader}/results.json")
    res = json.loads(Path(rp).read_text())
    fr = res["frame"]; rows = np.array(fr["rows_e_md_e_apical_e_buccal"])
    frame = geom.LocalFrame(fr["origin_SAL_mm"], rows[0], rows[1], rows[2], fr["offset_mm"])
    z_min, z_max = res["search_band_Z_mm"]
    prof = res["new_bone_md_profile"]
    sub_full = res["volume"]["implant_in_envelope_mm3"]
    excluded = set(base.get("exclude_control_windows") or [])
    reported = res["volume"]["PRIMARY_net_bone_gain_mm3"]
    depth = contained_depth(case)
    assert depth in SWEEP.DEPTHS, depth

    y_hi = frame.offset[1] + max(SWEEP.DEPTHS) + SWEEP.PAD_Y
    KEEP = geom.window_box({"X_mm": [-40.0, 60.0], "Y_mm": [frame.offset[1], y_hi]}, {},
                           z_min, z_max, pad_x_mm=0.0, pad_y_mm=0.0)
    pre, vx = geom.load_volume(f"analysis/{case}/pre_cal.nii.gz")
    post, _ = geom.load_volume(f"cbct_blinded/{case}/post_reg.nii.gz")
    pf, qf = geom.bone_field(pre), geom.bone_field(post)
    del pre, post
    Vp, Fp = geom.iso_surface(pf, vx); P = geom.surface_points_local(Vp, Fp, frame, KEEP)
    del Vp, Fp, pf
    Vq, Fq = geom.iso_surface(qf, vx); Q = geom.surface_points_local(Vq, Fq, frame, KEEP)
    del Vq, Fq, qf
    print(f"  {case}/{reader}: surfaces P={len(P):,} Q={len(Q):,}  peak RSS {rss_gb():.2f} GB",
          flush=True)

    w = WIDTH
    roi_w = landmarks.standard_roi(w, 10.0, offset=frame.offset, anchor="preop_anatomy")
    ctrl = landmarks.control_windows_clean(roi_w, prof, gap_mm=SWEEP.GAP, frac=SWEEP.FRAC,
                                           md_positive_is=frame.md_positive_is)
    place = ctrl.pop("_placement"); ctrl.pop("_rule")
    ctrl_ok = all(m.get("clean") for m in place.values())
    cells, fields, bgs = {}, {}, {}
    for dep in SWEEP.DEPTHS:
        roi = landmarks.standard_roi(w, dep, offset=frame.offset, anchor="preop_anatomy")
        bg, n_ok = [], 0
        for nm, xr in ctrl.items():
            if nm in excluded:
                continue
            b = V.measure_window(P, Q, xr, roi["Y_mm"], SWEEP.STEP, z_min, z_max, keep_box=KEEP)
            if b.get("cells", 0) < 200:
                continue
            bo = b["median_thickness_mm"]
            if abs(bo) > SWEEP.TOL:
                continue
            bg.append(bo); n_ok += 1
        background = float(np.mean(bg)) if bg else None
        m = V.measure_window(P, Q, roi["X_mm"], roi["Y_mm"], SWEEP.STEP, z_min, z_max,
                             background=(background or 0.0), keep_box=KEEP)
        if m.get("cells", 0) < 100:
            continue
        frac_area = (w * dep) / 100.0
        cells[dep] = {
            "roi_w": w, "roi_d": dep,
            "gross_mm3": round(m["gain_mm3"], 1),
            "net_mm3": round(m["gain_mm3"] - sub_full * frac_area, 1),
            "coverage": round(m.get("coverage", float("nan")), 4),
            "area_mm2": round(m.get("area_mm2", float("nan")), 2),
            "mean_thickness_mm": round(m.get("mean_thickness_mm", float("nan")), 3),
            "background_mm": None if background is None else round(background, 3),
            "n_controls": n_ok, "controls_placeable": bool(ctrl_ok),
        }
        fields[dep] = (m["_Hpre"], m["_Hpost"], m["_x_edges"], m["_y_edges"])
        bgs[dep] = background or 0.0
    del P, Q

    checks = {}
    sweep_path = Path(f"analysis/{case}/roi_extent_sweep{'' if reader == 'A' else '_' + reader}.json")
    if sweep_path.exists():
        stored = {g["roi_d"]: g for g in json.loads(sweep_path.read_text())["grid"]
                  if g["roi_w"] == w}
        for dep, c in cells.items():
            assert c == stored[dep], (case, reader, dep, c, stored[dep])
        checks["width10_column_equals_stored_sweep"] = {"file": str(sweep_path),
                                                        "cells_compared": len(cells),
                                                        "all_equal": True}
    else:
        checks["width10_column_equals_stored_sweep"] = {
            "file": str(sweep_path), "note": ""}
    ref = cells[10.0]
    vol = res["volume"]
    frozen = json.loads(FROZEN.read_text())["cases"][case][reader]
    assert ref["net_mm3"] == reported, (ref["net_mm3"], reported)
    assert ref["net_mm3"] == frozen["net_mm3"], (ref["net_mm3"], frozen["net_mm3"])
    assert ref["gross_mm3"] == vol["gain_before_implant_subtraction_mm3"]
    assert ref["background_mm"] == vol["background_subtracted_mm"]
    assert ref["area_mm2"] == vol["projected_area_mm2"]
    assert ref["mean_thickness_mm"] == vol["mean_thickness_mm"]
    checks["cell_10x10_equals_frozen"] = {
        "net_mm3": [ref["net_mm3"], reported, frozen["net_mm3"]],
        "gross_mm3": [ref["gross_mm3"], vol["gain_before_implant_subtraction_mm3"]],
        "background_mm": [ref["background_mm"], vol["background_subtracted_mm"]],
        "area_mm2": [ref["area_mm2"], vol["projected_area_mm2"]],
        "mean_thickness_mm": [ref["mean_thickness_mm"], vol["mean_thickness_mm"]],
        "all_equal": True,
        "precision_note": ""}
    print(f"  {case}/{reader}: (10,10) = {ref['net_mm3']} == frozen {frozen['net_mm3']}  OK",
          flush=True)

    def gain_with(dep, background):
        m = V.measure_window(None, None, [0, 0], [0, 0], SWEEP.STEP, z_min, z_max,
                             background=background, fields=fields[dep])
        return round(m["gain_mm3"] - sub_full * (w * dep) / 100.0, 1)

    sec = cells[depth]
    decomp = {
        "net_at_10_bg0_mm3": gain_with(10.0, 0.0),
        f"net_at_{depth:g}_bg0_mm3": gain_with(depth, 0.0),
        f"net_at_{depth:g}_bg_held_at_10x10_value_mm3": gain_with(depth, bgs[10.0]),
        "background_10_mm": cells[10.0]["background_mm"],
        f"background_{depth:g}_mm": sec["background_mm"],
    }
    out = {"case": case, "reader": reader,
           "contained_depth_mm": depth,
           "contained_depth_source": str(CONTAINMENT),
           "primary_10x10_mm3": ref["net_mm3"],
           "secondary_10xD_mm3": sec["net_mm3"],
           "secondary_minus_primary_mm3": round(sec["net_mm3"] - ref["net_mm3"], 1),
           "secondary_cell": sec, "primary_cell": ref,
           "width10_column": [cells[k] for k in sorted(cells)],
           "r6_decomposition": decomp,
           "checks": checks,
           "implant_in_envelope_mm3_full_window": sub_full,
           "peak_rss_gb": round(rss_gb(), 2)}
    p = Path(f"analysis/{case}/secondary_contained_depth_{reader}.json")
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"  written {p}   peak RSS {rss_gb():.2f} GB", flush=True)


def collect():
    cont = json.loads(CONTAINMENT.read_text())["cases"]
    table, per = [], {}
    for case in ["S1", "S2", "S3", "S4"]:
        for reader in ["A", "B"]:
            p = Path(f"analysis/{case}/secondary_contained_depth_{reader}.json")
            if not p.exists():
                table.append({"case": case, "reader": reader, "status": "NOT RUN"})
                continue
            o = json.loads(p.read_text())
            per[f"{case}_{reader}"] = o
            row = {"case": case, "reader": reader,
                   "contained_depth_mm": o["contained_depth_mm"],
                   "primary_10x10_mm3": o["primary_10x10_mm3"],
                   "secondary_10xD_mm3": o["secondary_10xD_mm3"],
                   "secondary_minus_primary_mm3": o["secondary_minus_primary_mm3"],
                   "background_10_mm": o["r6_decomposition"]["background_10_mm"],
                   "background_D_mm": o["secondary_cell"]["background_mm"],
                   "coverage_D": o["secondary_cell"]["coverage"],
                   "n_controls_D": o["secondary_cell"]["n_controls"],
                   "frozen_reproduced": o["checks"]["cell_10x10_equals_frozen"]["all_equal"]}
            if case == "S4" and reader == "A":
                naive = round(o["primary_10x10_mm3"] + cont["S4"]["gain_below_window_mm3"], 1)
                d = o["r6_decomposition"]; D = o["contained_depth_mm"]
                row["naive_primary_plus_null_excess_mm3"] = naive
                row["estimator_minus_naive_mm3"] = round(o["secondary_10xD_mm3"] - naive, 1)
                row["uncorrected_increment_10_to_D_mm3"] = round(
                    d[f"net_at_{D:g}_bg0_mm3"] - d["net_at_10_bg0_mm3"], 1)
                row["increment_with_background_held_mm3"] = round(
                    d[f"net_at_{D:g}_bg_held_at_10x10_value_mm3"] - o["primary_10x10_mm3"], 1)
                row["part_due_to_background_moving_mm3"] = round(
                    o["secondary_10xD_mm3"] - d[f"net_at_{D:g}_bg_held_at_10x10_value_mm3"], 1)
            table.append(row)
    result = {
        "_what": "",
        "_script": "analysis/secondary_contained_depth.py",
        "_depth_rule": "",
        "_reader_B_note": "",
        "_r6_note": "",
        "table": table, "per_run": per}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {OUT}")
    for r in table:
        print(r)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--collect" in argv:
        collect()
        sys.exit(0)
    rd = "A"
    if "--reader" in argv:
        i = argv.index("--reader")
        rd = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    for c in [x for x in argv if not x.startswith("-")]:
        run(c, rd)

