#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import geom, volume as V

STEP = 0.20
BANDS = [(0, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20)]
MAXD = 20.0
TILE_STEP = 2.0


def run(case):
    res = json.loads(Path(f"analysis/{case}/results.json").read_text())
    frame = geom.LocalFrame.from_dict(res["frame"])
    z_min, z_max = res["search_band_Z_mm"]
    y0, x0 = frame.offset[1], frame.offset[0]
    roi = res["roi"]["X_mm"]
    w = roi[1] - roi[0]
    prof = res["new_bone_md_profile"]
    px = np.array(prof["x_mm"], float)
    pv = np.array(prof["new_bone_mm3_per_slab"], float)
    thr = res["graft_extent_SECONDARY"]["threshold_mm3_per_mm"]

    KEEP = geom.window_box({"X_mm": [-40.0, 60.0], "Y_mm": [y0, y0 + MAXD + 3.0]}, {},
                           z_min, z_max, pad_x_mm=0.0, pad_y_mm=0.0)
    pre, vx = geom.load_volume(f"analysis/{case}/pre_cal.nii.gz")
    post, _ = geom.load_volume(f"cbct_blinded/{case}/post_reg.nii.gz")

    metal = {}
    for tag, vol in (("pre", pre), ("post", post)):
        idx = np.argwhere(vol > geom.METAL_LEVEL)
        if len(idx) == 0:
            metal[tag] = None
            continue
        loc = frame.to_local(idx * vx)
        m = ((loc[:, 0] > roi[0] - 15) & (loc[:, 0] < roi[1] + 15)
             & (loc[:, 1] > y0 - 2) & (loc[:, 1] < y0 + MAXD + 3))
        metal[tag] = None if not m.any() else {
            "n_voxels_near_site": int(m.sum()),
            "max_buccal_Z_mm": round(float(loc[m, 2].max()), 2),
            "max_depth_mm": round(float((loc[m, 1] - y0).max()), 2)}

    pf, qf = geom.bone_field(pre), geom.bone_field(post)
    del pre, post
    Vp, Fp = geom.iso_surface(pf, vx); P = geom.surface_points_local(Vp, Fp, frame, KEEP)
    del Vp, Fp, pf
    Vq, Fq = geom.iso_surface(qf, vx); Q = geom.surface_points_local(Vq, Fq, frame, KEEP)
    del Vq, Fq, qf

    def bands_for(xr):
        o = {}
        for lo, hi in BANDS:
            m = V.measure_window(P, Q, xr, [y0 + lo, y0 + hi], STEP, z_min, z_max, keep_box=KEEP)
            if m.get("cells", 0) < 100:
                o[f"{lo}-{hi}"] = None
            else:
                o[f"{lo}-{hi}"] = {"gain_mm3": m["gain_mm3"],
                                   "density_mm3_per_mm2": round(m["gain_mm3"] / m["area_mm2"], 4),
                                   "mean_thickness_mm": m["mean_thickness_mm"],
                                   "coverage": m["coverage"]}
        return o

    roi_bands = bands_for(roi)

    lo_x, hi_x = -25.0, 55.0
    nulls = []
    x = lo_x
    while x + w <= hi_x:
        xr = [x, x + w]
        overlaps = not (xr[1] <= roi[0] - 2.0 or xr[0] >= roi[1] + 2.0)
        sel = (px >= xr[0]) & (px < xr[1])
        clean = bool(sel.any() and np.nanmean(pv[sel]) <= thr)
        if clean and not overlaps:
            b = bands_for(xr)
            if b["0-10"] is not None:
                nulls.append({"x_mm": [round(xr[0], 1), round(xr[1], 1)], "bands": b})
        x += TILE_STEP
    del P, Q

    print(f"\n===== {case} =====  {len(nulls)} augmentation-free null windows tiled along the arch")
    print(f"{'band (mm deep)':>16}{'ROI density':>13}{'null median':>13}{'null IQR':>20}"
          f"{'ROI percentile':>16}   verdict")
    out_bands = {}
    for lo, hi in BANDS:
        k = f"{lo}-{hi}"
        r = roi_bands.get(k)
        vals = [n["bands"][k]["density_mm3_per_mm2"] for n in nulls
                if n["bands"].get(k) is not None]
        if r is None or not vals:
            continue
        v = np.array(vals)
        pct = float((v < r["density_mm3_per_mm2"]).mean() * 100)
        verdict = ("above the null" if pct >= 95 else
                   "INSIDE THE NULL - not distinguishable from undisturbed bone")
        print(f"{k:>16}{r['density_mm3_per_mm2']:>13.3f}{np.median(v):>13.3f}"
              f"{f'{np.percentile(v,25):.3f}-{np.percentile(v,75):.3f}':>20}{pct:>15.0f}%   {verdict}")
        out_bands[k] = {"roi_density": r["density_mm3_per_mm2"],
                        "roi_gain_mm3": r["gain_mm3"],
                        "roi_mean_thickness_mm": r["mean_thickness_mm"],
                        "null_n": len(v), "null_median": round(float(np.median(v)), 4),
                        "null_iqr": [round(float(np.percentile(v, 25)), 4),
                                     round(float(np.percentile(v, 75)), 4)],
                        "null_max": round(float(v.max()), 4),
                        "roi_percentile_in_null": round(pct, 1),
                        "distinguishable_from_undisturbed_bone": bool(pct >= 95)}

    cum, cum_raw, dcur = 0.0, 0.0, {}
    for lo, hi in BANDS:
        k = f"{lo}-{hi}"
        if k not in out_bands:
            continue
        b = out_bands[k]
        area = (roi[1] - roi[0]) * (hi - lo)
        cum_raw += b["roi_gain_mm3"]
        cum += max(0.0, (b["roi_density"] - b["null_median"]) * area)
        dcur[str(hi)] = {"raw_cumulative_mm3": round(cum_raw, 1),
                         "null_corrected_cumulative_mm3": round(cum, 1)}

    print(f"\n  cumulative gain with depth (before implant subtraction):")
    for d, v in dcur.items():
        print(f"    to {d:>2} mm:  raw {v['raw_cumulative_mm3']:>7.1f}   "
              f"null-corrected {v['null_corrected_cumulative_mm3']:>7.1f} mm3")
    print(f"\n  titanium-level voxels near the site: "
          f"pre {metal['pre']}  post {metal['post']}")

    out = {"case": case, "_what": "",
           "_regenerates": (""),
           "n_null_windows": len(nulls), "tile_step_mm": TILE_STEP, "window_width_mm": w,
           "roi_bands": roi_bands, "band_comparison": out_bands,
           "depth_cumulative": dcur,
           "metal_extent_near_site": metal,
           "null_windows": nulls}
    Path(f"analysis/{case}/null_window_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwritten analysis/{case}/null_window_check.json")


if __name__ == "__main__":
    for c in ([x for x in sys.argv[1:] if not x.startswith("-")] or ["S4"]):
        run(c)

