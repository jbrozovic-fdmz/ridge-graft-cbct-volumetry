from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import geom


def _edges(lo, hi, step):
    return np.arange(lo, hi + 1e-9, step)


def height_fields(pre_pts, post_pts, x_range, y_range, step, z_min=-np.inf, z_max=np.inf,
                  mode="top_mean", spike_tol_mm=2.0, keep_box=None):
    xe, ye = _edges(*x_range, step), _edges(*y_range, step)
    if keep_box is not None:
        x0, x1, y0, y1, z0, z1 = keep_box
        if not (xe[0] >= x0 and xe[-1] <= x1 and ye[0] >= y0 and ye[-1] <= y1
                and z_min >= z0 and z_max <= z1):
            raise SystemExit(
                f"ABORT: measurement window X[{xe[0]},{xe[-1]}] Y[{ye[0]},{ye[-1]}] "
                f"Z[{z_min},{z_max}] falls outside the point-cloud pre-filter box {keep_box}. "
                f"The clouds were filtered to that box in step 4, so this window would read a "
                f"truncated surface. Widen geom.window_box() rather than removing this check.")
    Hpre = geom.height_field(pre_pts, xe, ye, z_min, z_max, mode=mode, spike_tol_mm=spike_tol_mm)
    Hpost = geom.height_field(post_pts, xe, ye, z_min, z_max, mode=mode, spike_tol_mm=spike_tol_mm)
    return Hpre, Hpost, xe, ye


def measure_window(pre_pts, post_pts, x_range, y_range, step=0.2, z_min=-np.inf,
                   z_max=np.inf, background=0.0, mode="top_mean", spike_tol_mm=2.0,
                   keep_box=None, fields=None):
    if fields is not None:
        Hpre, Hpost, xe, ye = fields
    else:
        if pre_pts is None or post_pts is None:
            raise SystemExit("ABORT: measure_window needs either point clouds or `fields`; "
                             "it was given neither, and would have measured nothing.")
        Hpre, Hpost, xe, ye = height_fields(pre_pts, post_pts, x_range, y_range, step, z_min,
                                            z_max, mode, spike_tol_mm, keep_box=keep_box)
    thick = Hpost - Hpre - background
    ok = np.isfinite(thick)
    cell = step * step
    if not ok.any():
        return {"cells": 0, "coverage": 0.0}
    t = thick[ok]
    area = float(ok.sum() * cell)
    return {"cells": int(ok.sum()),
            "coverage": round(float(ok.mean()), 3),
            "area_mm2": round(area, 2),
            "gain_mm3": round(float(t[t > 0].sum() * cell), 1),
            "loss_mm3": round(float(-t[t < 0].sum() * cell), 1),
            "net_mm3": round(float(t.sum() * cell), 1),
            "mean_thickness_mm": round(float(t.mean()), 3),
            "median_thickness_mm": round(float(np.median(t)), 3),
            "max_thickness_mm": round(float(t.max()), 2),
            "p95_thickness_mm": round(float(np.percentile(t, 95)), 2),
            "volume_to_surface_ratio_mm3_per_mm2": round(float(t.sum() * cell / area), 3),
            "_thickness": thick, "_x_edges": xe, "_y_edges": ye,
            "_Hpre": Hpre, "_Hpost": Hpost}


def implant_volume_in_envelope(implant_voxels_SAL, frame, Hpre, Hpost, xe, ye, voxel_mm):
    vox = voxel_mm ** 3
    if implant_voxels_SAL is None or not len(implant_voxels_SAL):
        return {"implant_in_envelope_mm3": 0.0, "implant_total_mm3": 0.0, "n_voxels": 0,
                "note": ""}
    centres = np.asarray(implant_voxels_SAL, float) * voxel_mm
    P = frame.to_local(centres)
    ny, nx = Hpre.shape
    ix = np.digitize(P[:, 0], xe) - 1
    iy = np.digitize(P[:, 1], ye) - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    total = round(len(centres) * vox, 1)
    if not inside.any():
        return {"implant_in_envelope_mm3": 0.0, "implant_total_mm3": total, "n_voxels": 0}

    lo = Hpre[iy[inside], ix[inside]]
    hi = Hpost[iy[inside], ix[inside]]
    z = P[inside, 2]
    half = voxel_mm / 2.0
    frac = np.clip(np.minimum(hi, z + half) - np.maximum(lo, z - half), 0.0, voxel_mm) / voxel_mm
    frac = np.where(np.isfinite(lo) & np.isfinite(hi), frac, 0.0)
    return {"implant_in_envelope_mm3": round(float(frac.sum()) * vox, 1),
            "implant_total_mm3": total,
            "implant_in_roi_mm3": round(float(inside.sum()) * vox, 1),
            "n_voxels": int(inside.sum()),
            "method": "",
            "note": ""}


def boolean_volume(pre_field, post_field, frame, roi, voxel_mm, implant_mask=None,
                   levels=(350.0, 450.0, 550.0, 750.0), z_min=3.0, z_max=27.0, opening=True):
    shape = np.array(pre_field.shape)
    xs = np.arange(roi["X_mm"][0], roi["X_mm"][1], voxel_mm)
    ys = np.arange(roi["Y_mm"][0], roi["Y_mm"][1], voxel_mm)
    zs = np.arange(z_min, z_max, voxel_mm)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Q = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    P = frame.to_anatomical(Q) / voxel_mm
    ok = np.all((P >= 0) & (P < shape - 1), axis=1)
    out = {}
    pre_s = np.full(len(P), -1000.0)
    post_s = np.full(len(P), -1000.0)
    pre_s[ok] = ndi.map_coordinates(pre_field, P[ok].T, order=1)
    post_s[ok] = ndi.map_coordinates(post_field, P[ok].T, order=1)
    imp_s = np.zeros(len(P), bool)
    if implant_mask is not None:
        dil = ndi.binary_dilation(implant_mask, np.ones((3, 3, 3)), iterations=3)
        del implant_mask
        imp_s[ok] = ndi.map_coordinates(dil.view(np.uint8), P[ok].T, order=0) > 0
        del dil
    cell = voxel_mm ** 3
    for lv in levels:
        gained = ((post_s > lv) & (pre_s <= lv) & ~imp_s).reshape(X.shape)
        lost = ((pre_s > lv) & (post_s <= lv) & ~imp_s).reshape(X.shape)
        rec = {"gained_raw_mm3": round(float(gained.sum()) * cell, 1),
               "lost_raw_mm3": round(float(lost.sum()) * cell, 1),
               "net_mm3": round(float(gained.sum() - lost.sum()) * cell, 1)}
        if opening:
            g = ndi.binary_opening(gained, np.ones((3, 3, 3)))
            lab, n = ndi.label(g)
            big = float(ndi.sum(g, lab, range(1, n + 1)).max()) if n else 0.0
            rec["gained_opened_mm3"] = round(float(g.sum()) * cell, 1)
            rec["largest_component_mm3"] = round(big * cell, 1)
            rec["n_components"] = int(n)
        out[str(int(lv))] = rec
    out["coverage"] = round(float(ok.mean()), 3)
    out["band_Z_mm"] = [round(z_min, 2), round(z_max, 2)]
    out["_method"] = ("")
    return out


def new_bone_md_profile(pre_field, post_field, frame, voxel_mm, z_min, z_max,
                        x_range=(-25.0, 55.0), y_range=(15.0, 25.0), step_mm=1.0,
                        level=None):
    from . import geom as _geom
    if level is None:
        level = _geom.ISO_LEVEL
    xs = np.arange(x_range[0], x_range[1], step_mm)
    ys = np.arange(y_range[0], y_range[1], voxel_mm)
    zs = np.arange(z_min, z_max, voxel_mm)
    shape = np.array(pre_field.shape)
    cell = step_mm * voxel_mm * voxel_mm
    out = []
    for x0 in xs:
        X, Y, Z = np.meshgrid(np.array([x0 + 0.5 * step_mm]), ys, zs, indexing="ij")
        Q = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        P = frame.to_anatomical(Q) / voxel_mm
        ok = np.all((P >= 0) & (P < shape - 1), axis=1)
        if not ok.any():
            out.append((float(x0 + 0.5 * step_mm), 0.0, 0.0))
            continue
        a = ndi.map_coordinates(pre_field, P[ok].T, order=1)
        b = ndi.map_coordinates(post_field, P[ok].T, order=1)
        gained = float(((b > level) & (a <= level)).sum()) * cell
        out.append((float(x0 + 0.5 * step_mm), round(gained, 2), round(float(ok.mean()), 3)))
    return {"step_mm": step_mm, "x_mm": [o[0] for o in out],
            "new_bone_mm3_per_slab": [o[1] for o in out],
            "coverage": [o[2] for o in out],
            "_method": ("")}


def _profile_baseline(x, v, peak_i, far_mm=20.0):
    far = np.abs(x - x[peak_i]) > far_mm
    return float(np.median(v[far])) if far.sum() >= 5 else float(np.percentile(v, 25))


def graft_extent(profile, frac=0.20, far_mm=20.0):
    x = np.asarray(profile["x_mm"], float)
    v = np.asarray(profile["new_bone_mm3_per_slab"], float)
    if not len(v) or v.max() <= 0:
        return {"extent_mm": 0.0, "peak_x_mm": None, "total_mm3": 0.0, "baseline_mm3_per_mm": 0.0}
    pk = int(np.argmax(v))
    base = _profile_baseline(x, v, pk, far_mm)
    thr = base + frac * (v[pk] - base)
    lo = pk
    while lo > 0 and v[lo - 1] > thr:
        lo -= 1
    hi = pk
    while hi < len(v) - 1 and v[hi + 1] > thr:
        hi += 1
    excess = float(np.clip(v[lo:hi + 1] - base, 0, None).sum())
    return {"extent_mm": round(float(x[hi] - x[lo] + profile["step_mm"]), 1),
            "x_from_mm": round(float(x[lo]), 1), "x_to_mm": round(float(x[hi]), 1),
            "peak_x_mm": round(float(x[pk]), 1),
            "peak_mm3_per_mm": round(float(v[pk]), 2),
            "baseline_mm3_per_mm": round(base, 2),
            "threshold_mm3_per_mm": round(float(thr), 2),
            "total_above_baseline_mm3": round(excess, 1),
            "threshold_rule": f"baseline + {frac:.0%} of (peak - baseline)",
            "_note": ("")}


def gain_profile(thickness, xe, ye, roi):
    depth = (ye[:-1] + np.diff(ye) / 2) - roi["Y_mm"][0]
    along = (xe[:-1] + np.diff(xe) / 2) - (roi["X_mm"][0] + roi["size_md_mm"] / 2)
    with np.errstate(all="ignore"):
        d_mean = np.nanmean(thickness, axis=1)
        d_max = np.nanmax(thickness, axis=1)
        a_mean = np.nanmean(thickness, axis=0)
        a_max = np.nanmax(thickness, axis=0)

    def at(d):
        i = int(np.argmin(np.abs(depth - d)))
        return [round(float(d_mean[i]), 2), round(float(d_max[i]), 2)]

    ipk = int(np.nanargmax(d_mean))
    return {"_depth": depth, "_depth_mean": d_mean, "_depth_max": d_max,
            "_along": along, "_along_mean": a_mean, "_along_max": a_max,
            "levels_mm_apical_to_platform": {"_format": "[mean, max] mm",
                                             "1": at(1), "2": at(2), "3": at(3),
                                             "5": at(5), "7": at(7)},
            "peak": {"depth_mm": round(float(depth[ipk]), 1),
                     "mean_mm": round(float(d_mean[ipk]), 2),
                     "max_mm": round(float(d_max[ipk]), 2)},
            "overall_mean_mm": round(float(np.nanmean(thickness)), 2),
            "overall_max_mm": round(float(np.nanmax(thickness)), 2)}

