#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import geom, landmarks, volume as V

WIDTHS = (8.0, 10.0, 12.0)
DEPTHS = (6.0, 8.0, 10.0)
STEPS = (0.15, 0.20, 0.30)
FRACS = (0.15, 0.20, 0.30)
GAPS = (2.0, 4.0)
TOL = 0.90


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

    pre, vx = geom.load_volume(f"analysis/{case}/pre_cal.nii.gz")
    post, _ = geom.load_volume(f"cbct_blinded/{case}/post_reg.nii.gz")
    pf, qf = geom.bone_field(pre), geom.bone_field(post)
    del pre, post
    KEEP = geom.window_box({"X_mm": [-40.0, 60.0], "Y_mm": [15.0, 30.0]}, {},
                           z_min, z_max, pad_x_mm=0.0, pad_y_mm=0.0)
    Vp, Fp = geom.iso_surface(pf, vx); P = geom.surface_points_local(Vp, Fp, frame, KEEP)
    del Vp, Fp, pf
    Vq, Fq = geom.iso_surface(qf, vx); Q = geom.surface_points_local(Vq, Fq, frame, KEEP)
    del Vq, Fq, qf

    specs = []
    for w in WIDTHS:
        for dep in DEPTHS:
            roi = landmarks.standard_roi(w, dep, offset=frame.offset, anchor="preop_anatomy")
            for frac in FRACS:
                for gap in GAPS:
                    ctrl = landmarks.control_windows_clean(
                        roi, prof, gap_mm=gap, frac=frac,
                        md_positive_is=frame.md_positive_is)
                    place = ctrl.pop("_placement"); ctrl.pop("_rule")
                    if any(not m.get("clean") for m in place.values()):
                        continue
                    bg, ok = [], True
                    for nm, xr in ctrl.items():
                        if nm in excluded:
                            continue
                        b = V.measure_window(P, Q, xr, roi["Y_mm"], 0.20, z_min, z_max, keep_box=KEEP)
                        l = V.measure_window(P, Q, xr, roi["Y_mm"], 0.20, z_min, z_max,
                                             mode="bottom_mean", keep_box=KEEP)
                        if b.get("cells", 0) < 200:
                            ok = False; break
                        bo = b["median_thickness_mm"]
                        lo_ = -l["median_thickness_mm"] if l.get("cells", 0) >= 200 else None
                        if lo_ is not None:
                            if abs((bo - lo_) / 2) > TOL or abs((bo + lo_) / 2) > TOL:
                                continue
                        if abs(bo) > TOL:
                            ok = False; break
                        bg.append(bo)
                    if not ok or not bg:
                        continue
                    background = float(np.mean(bg))
                    for step in STEPS:
                        m = V.measure_window(P, Q, roi["X_mm"], roi["Y_mm"], step, z_min, z_max,
                                             background=background, keep_box=KEEP)
                        if m.get("cells", 0) < 100:
                            continue
                        frac_area = (w * dep) / 100.0
                        specs.append({"roi_w": w, "roi_d": dep, "step": step, "frac": frac,
                                      "gap": gap, "background_mm": round(background, 3),
                                      "n_controls": len(bg),
                                      "gross_mm3": m["gain_mm3"],
                                      "net_mm3": round(m["gain_mm3"] - sub_full * frac_area, 1)})
    del P, Q
    nets = np.array([s["net_mm3"] for s in specs]) if specs else np.array([])
    rep = res["volume"]["PRIMARY_net_bone_gain_mm3"]
    out = {"case": case, "reader": reader, "reported_mm3": rep,
           "n_specifications": len(specs),
           "min_mm3": None if not len(nets) else round(float(nets.min()), 1),
           "max_mm3": None if not len(nets) else round(float(nets.max()), 1),
           "median_mm3": None if not len(nets) else round(float(np.median(nets)), 1),
           "iqr_mm3": None if not len(nets) else [round(float(np.percentile(nets, 25)), 1),
                                                  round(float(np.percentile(nets, 75)), 1)],
           "reported_percentile": None if not len(nets) else round(float((nets < rep).mean() * 100), 1),
           "threshold_sweep_mm3": res["volume"]["threshold_sensitivity_mm3"],
           "_dimensions": {"roi_width_mm": WIDTHS, "roi_depth_mm": DEPTHS, "raster_step_mm": STEPS,
                           "control_frac": FRACS, "control_gap_mm": GAPS},
           "_admissibility": (""),
           "specifications": specs}
    Path(f"analysis/{case}").mkdir(exist_ok=True)
    tag = "" if reader == "A" else f"_{reader}"
    Path(f"analysis/{case}/specification_curve{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{case} ({reader}): reported {rep} | {len(specs)} specs | "
          f"{out['min_mm3']}–{out['max_mm3']} | median {out['median_mm3']} "
          f"| IQR {out['iqr_mm3']} | reported sits at the {out['reported_percentile']}th pct")


if __name__ == "__main__":
    argv = sys.argv[1:]
    rd = "A"
    if "--reader" in argv:
        i = argv.index("--reader")
        rd = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    a = [x for x in argv if not x.startswith("-")]
    for c in (a or ["S1", "S2", "S3", "S4"]):
        run(c, rd)

