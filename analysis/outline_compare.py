#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import geom, plates
from pipeline.arch import ArchCurve

GRID = 0.05
WINDOW_DEPTH = 10.0
trapz = getattr(np, "trapezoid", None) or np.trapz


def run(case):
    d = json.loads(Path(f"cases/{case}_outlines.json").read_text())
    res = json.loads(Path(f"analysis/{case}/results.json").read_text())
    frame = geom.LocalFrame.from_dict(res["frame"])
    aj = json.loads(Path(f"cases/{case}_arch.json").read_text())
    arch = ArchCurve(np.array(aj["arch_points_AL_mm"], float))
    arch.orient_normal_outward(arch.points.mean(axis=0))
    eb = np.array(res["frame"]["rows_e_md_e_apical_e_buccal"][2], float)
    yv = float(np.mean(res["roi"]["Y_mm"]))
    st0 = arch.nearest_station(*frame.to_anatomical(
        np.array([[frame.offset[0], yv, frame.offset[2]]]))[0][1:])
    if float(arch.normal[st0, 0] * eb[1] + arch.normal[st0, 1] * eb[2]) < 0:
        arch.normal *= -1.0
    sgn = 1.0 if frame.jaw == "maxilla" else -1.0
    cs = d["pre_op_crest_S_mm"]
    half, above, below = d["half_width_mm"], d["section_above_mm"], d["section_below_mm"]

    pre_v, vx = geom.load_volume(f"analysis/{case}/pre_cal.nii.gz")
    post_v, _ = geom.load_volume(f"cbct_blinded/{case}/post_reg.nii.gz")

    rows = []
    for k, v in sorted(d["stations"].items(), key=lambda t: int(t[0])):
        if len(v["pre_vertices_us_mm"]) < 3 or len(v["post_vertices_us_mm"]) < 3:
            continue
        st = int(k)
        a, b = sorted((cs - sgn * above, cs + sgn * below))
        s_lo, s_hi = max(0.0, a), min((pre_v.shape[0] - 1) * vx, b)
        pre_img, ext, u, s_ax = arch.section(pre_v, vx, st, half_width_mm=half,
                                             s_range=(s_lo, s_hi))
        post_img, _, _, _ = arch.section(post_v, vx, st, half_width_mm=half,
                                         s_range=(s_lo, s_hi))
        pix = (ext[1] - ext[0]) / (post_img.shape[1] - 1)
        sig = geom.SMOOTH_SIGMA * vx / pix

        def _f(arr):
            return ndi.gaussian_filter(
                np.where(arr > geom.METAL_LEVEL, -1000.0, arr).astype(np.float32), sig)
        pb = plates._buccal_trace(_f(pre_img), u, s_ax)
        qb = plates._buccal_trace(_f(post_img), u, s_ax)

        ug = np.arange(ext[0], ext[1], GRID)
        sg = np.arange(ext[2], ext[3], GRID)
        U, S = np.meshgrid(ug, sg)
        pts = np.column_stack([U.ravel(), S.ravel()])
        cell = GRID * GRID
        depth = sgn * (S - cs)
        win = (depth >= 0) & (depth <= WINDOW_DEPTH)

        inP = MplPath(np.array(v["pre_vertices_us_mm"], float)).contains_points(pts).reshape(U.shape)
        inQ = MplPath(np.array(v["post_vertices_us_mm"], float)).contains_points(pts).reshape(U.shape)
        bucc = U >= 0.0
        roi = win & bucc
        man_win = ((inQ & roi).sum() - (inP & roi).sum()) * cell
        man_gain = (inQ & ~inP & roi).sum() * cell
        man_loss = (inP & ~inQ & roi).sum() * cell
        man_outside = ((inQ & ~roi).sum() - (inP & ~roi).sum()) * cell
        pal_win = ((inQ & win & ~bucc).sum() - (inP & win & ~bucc).sum()) * cell

        pbi = np.interp(sg, s_ax, np.where(np.isfinite(pb), pb, np.nan))
        qbi = np.interp(sg, s_ax, np.where(np.isfinite(qb), qb, np.nan))
        fin = np.isfinite(pbi)[:, None] & np.isfinite(qbi)[:, None]
        band = (U >= pbi[:, None]) & (U <= qbi[:, None]) & fin
        lossband = (U >= qbi[:, None]) & (U <= pbi[:, None]) & fin
        alg_full = band.sum() * cell
        alg_win = (band & win & bucc).sum() * cell
        alg_loss = (lossband & win & bucc).sum() * cell

        rows.append({"station": st, "arc_length_mm": v["arc_length_mm"],
                     "manual_net_roi_mm2": round(man_win, 2),
                     "manual_gain_roi_mm2": round(man_gain, 2),
                     "manual_loss_roi_mm2": round(man_loss, 2),
                     "algorithm_gain_roi_mm2": round(alg_win, 2),
                     "algorithm_loss_roi_mm2": round(alg_loss, 2),
                     "manual_palatal_in_depth_band_mm2": round(pal_win, 2),
                     "manual_outside_roi_mm2": round(man_outside, 2),
                     "algorithm_gain_full_section_mm2": round(alg_full, 2)})

    rep = res["volume"]["PRIMARY_net_bone_gain_mm3"]
    mw = np.array([r["manual_net_roi_mm2"] for r in rows])
    aw = np.array([r["algorithm_gain_roi_mm2"] for r in rows])
    mg = np.array([r["manual_gain_roi_mm2"] for r in rows])
    ml = np.array([r["manual_loss_roi_mm2"] for r in rows])
    al = np.array([r["algorithm_loss_roi_mm2"] for r in rows])
    pal = np.array([r["manual_palatal_in_depth_band_mm2"] for r in rows])
    outs = np.array([r["manual_outside_roi_mm2"] for r in rows])
    arc = np.array([r["arc_length_mm"] for r in rows])

    print(f"\n===== {case} =====   reported (automatic, standardised window) {rep} mm3\n")
    print("  inside the standardised ROI (0-10 mm deep, buccal of the arch curve).")
    print("  GAIN-vs-GAIN is the comparison that matches the reported outcome.\n")
    print(f"{'stn':>5}{'arc':>7} |{'operator':>19} |{'algorithm':>19} |{'gain':>8}{'net':>8}")
    print(f"{'':>5}{'':>7} |{'gain':>9}{'loss':>10} |{'gain':>9}{'loss':>10} |"
          f"{'ratio':>8}{'ratio':>8}")
    for r, g_, l_, a_, al_ in zip(rows, mg, ml, aw, al):
        rg = g_ / a_ if a_ else float("nan")
        mn, an = g_ - l_, a_ - al_
        rn = mn / an if abs(an) > 1e-9 else float("nan")
        print(f"{r['station']:>5}{r['arc_length_mm']:>7.1f} |{g_:>9.1f}{l_:>10.1f} |"
              f"{a_:>9.1f}{al_:>10.1f} |{rg:>8.2f}{rn:>8.2f}")

    ratio = float(mg.sum() / aw.sum()) if aw.sum() else float("nan")
    per = mg / np.where(aw == 0, np.nan, aw)
    ratio_net = (float((mg - ml).sum() / (aw - al).sum())
                 if abs((aw - al).sum()) > 1e-9 else float("nan"))
    v_cal = rep * ratio
    v_manual_full = float(trapz(mg, arc)) - float(trapz(ml, arc))
    v_manual_win = float(trapz(mw, arc))
    print(f"\n  pooled GAIN ratio operator/algorithm   {ratio:.3f}"
          f"   (per-station {np.nanmin(per):.2f}-{np.nanmax(per):.2f}, SD {np.nanstd(per):.2f})")
    print(f"  pooled NET  ratio operator/algorithm   {ratio_net:.3f}")
    print(f"  CALIBRATED windowed volume  = {rep} x {ratio:.3f} = {v_cal:.1f} mm3"
          f"   ({100*(v_cal-rep)/rep:+.1f} % on the reported value)")
    print(f"  operator planimetry integrated over the {arc.max()-arc.min():.0f} mm marked span")
    print(f"      net inside the ROI               {v_manual_win:>7.1f} mm3")
    print(f"      (gain {float(trapz(mg,arc)):.1f}, loss {float(trapz(ml,arc)):.1f})")
    print(f"  EXCLUDED, as drawing noise rather than bone:")
    print(f"      palatal side within the depth band {float(trapz(pal,arc)):>7.1f} mm3")
    print(f"      everything outside the ROI         {float(trapz(outs,arc)):>7.1f} mm3")
    print(f"      -> the excluded terms are {abs(float(trapz(pal,arc)))+abs(float(trapz(outs,arc))):.0f} mm3, "
          f"{100*(abs(float(trapz(pal,arc)))+abs(float(trapz(outs,arc))))/max(v_manual_win,1e-9):.0f} % "
          f"of the ROI figure -- which is why they are not counted")

    out = {"case": case, "reported_windowed_mm3": rep, "stations": rows,
           "pooled_gain_ratio_operator_over_algorithm": round(ratio, 4),
           "pooled_net_ratio_operator_over_algorithm": round(ratio_net, 4),
           "algorithm_loss_roi_mm3": round(float(trapz(al, arc)), 1),
           "per_station_ratio": [None if not np.isfinite(x) else round(float(x), 3) for x in per],
           "calibrated_windowed_mm3": round(v_cal, 1),
           "calibration_shift_pct": round(100 * (v_cal - rep) / rep, 1),
           "operator_planimetry_marked_span_mm": round(float(arc.max() - arc.min()), 1),
           "operator_windowed_mm3": round(v_manual_win, 1),
           "operator_gain_only_mm3": round(float(trapz(mg, arc)), 1),
           "operator_loss_only_mm3": round(float(trapz(ml, arc)), 1),
           "excluded_palatal_mm3": round(float(trapz(pal, arc)), 1),
           "excluded_outside_roi_mm3": round(float(trapz(outs, arc)), 1),
           "_note": ("")}
    Path(f"analysis/{case}/outline_compare.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    for c in ([x for x in sys.argv[1:] if not x.startswith("-")] or ["S3", "S4"]):
        run(c)

