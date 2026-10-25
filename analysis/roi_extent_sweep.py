#!/usr/bin/env python3
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import geom, landmarks, volume as V

WIDTHS = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 24.0)
DEPTHS = (6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
STEP = 0.20
FRAC = 0.20
GAP = 2.0
TOL = 0.90
PAD_Y = 3.0


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def run(case, reader="A"):
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

    y_hi = frame.offset[1] + max(DEPTHS) + PAD_Y
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
    print(f"  surfaces loaded: P={len(P):,} Q={len(Q):,} pts   peak RSS {rss_gb():.2f} GB",
          flush=True)

    grid = {}
    for w in WIDTHS:
        roi_w = landmarks.standard_roi(w, 10.0, offset=frame.offset, anchor="preop_anatomy")
        ctrl = landmarks.control_windows_clean(roi_w, prof, gap_mm=GAP, frac=FRAC,
                                               md_positive_is=frame.md_positive_is)
        place = ctrl.pop("_placement"); ctrl.pop("_rule")
        ctrl_ok = all(m.get("clean") for m in place.values())
        for dep in DEPTHS:
            roi = landmarks.standard_roi(w, dep, offset=frame.offset, anchor="preop_anatomy")
            bg, n_ok = [], 0
            for nm, xr in ctrl.items():
                if nm in excluded:
                    continue
                b = V.measure_window(P, Q, xr, roi["Y_mm"], STEP, z_min, z_max, keep_box=KEEP)
                if b.get("cells", 0) < 200:
                    continue
                bo = b["median_thickness_mm"]
                if abs(bo) > TOL:
                    continue
                bg.append(bo); n_ok += 1
            background = float(np.mean(bg)) if bg else None
            m = V.measure_window(P, Q, roi["X_mm"], roi["Y_mm"], STEP, z_min, z_max,
                                 background=(background or 0.0), keep_box=KEEP)
            if m.get("cells", 0) < 100:
                continue
            frac_area = (w * dep) / 100.0
            grid[(w, dep)] = {
                "roi_w": w, "roi_d": dep,
                "gross_mm3": round(m["gain_mm3"], 1),
                "net_mm3": round(m["gain_mm3"] - sub_full * frac_area, 1),
                "coverage": round(m.get("coverage", float("nan")), 4),
                "area_mm2": round(m.get("area_mm2", float("nan")), 2),
                "mean_thickness_mm": round(m.get("mean_thickness_mm", float("nan")), 3),
                "background_mm": None if background is None else round(background, 3),
                "n_controls": n_ok, "controls_placeable": bool(ctrl_ok),
            }
        print(f"  width {w:>4.0f} mm done   peak RSS {rss_gb():.2f} GB", flush=True)

    ref = grid.get((10.0, 10.0))
    ok = ref is not None and abs(ref["net_mm3"] - reported) <= 0.15
    print(f"  (10,10) reproduces freeze: {ref and ref['net_mm3']} vs {reported}  -> "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)

    out = {
        "case": case, "reader": reader, "reported_10x10_mm3": reported,
        "regression_10x10": {"value": ref and ref["net_mm3"], "frozen": reported, "match": ok},
        "step_mm": STEP, "control_frac": FRAC, "control_gap_mm": GAP, "tolerance_mm": TOL,
        "widths_mm": list(WIDTHS), "depths_mm": list(DEPTHS),
        "peak_rss_gb": round(rss_gb(), 2),
        "grid": [grid[k] for k in sorted(grid)],
    }
    Path(f"analysis/{case}").mkdir(exist_ok=True)
    tag = "" if reader == "A" else f"_{reader}"
    Path(f"analysis/{case}/roi_extent_sweep{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{case}: {len(grid)} cells, peak RSS {rss_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    rd = "A"
    if "--reader" in argv:
        i = argv.index("--reader")
        rd = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    a = [x for x in argv if not x.startswith("-")]
    for c in (a or ["S1", "S2", "S3", "S4"]):
        print(f"=== {c} ({rd}) ===", flush=True)
        run(c, rd)

