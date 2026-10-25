from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import geom


def _cylinder_radius(pts, centroid, axis, proj):
    perp = (pts - centroid) - np.outer(proj, axis)
    return float(np.sqrt(2.0 * np.mean((perp ** 2).sum(axis=1))))


def find_implant(post_vol, pre_vol, voxel_mm, min_height_mm=6.0, min_new_frac=0.9):
    dense = geom.metal_mask(post_vol)
    lbl, n = ndi.label(dense)
    if n == 0:
        return None
    cands = []
    for i in range(1, n + 1):
        idx = np.argwhere(lbl == i)
        if len(idx) < 50:
            continue
        extent = (idx.max(axis=0) - idx.min(axis=0)) * voxel_mm
        height = float(extent.max())
        new_frac = 1.0 - float(np.mean(geom.metal_mask(pre_vol)[tuple(idx.T)]))
        cands.append(dict(label=int(i), n_voxels=int(len(idx)), height_mm=round(height, 2),
                          new_in_post_frac=round(new_frac, 3),
                          bbox_min=idx.min(axis=0).tolist(), bbox_max=idx.max(axis=0).tolist(),
                          idx=idx))
    keep = [c for c in cands if c["height_mm"] >= min_height_mm and c["new_in_post_frac"] >= min_new_frac]
    if not keep:
        return {"present": False,
                "candidates": [{k: v for k, v in c.items() if k != "idx"} for c in cands]}
    if len(keep) > 1:
        keep.sort(key=lambda c: -c["n_voxels"])
    best = keep[0]

    idx = best["idx"]
    pts = idx.astype(float) * voxel_mm
    centroid = pts.mean(axis=0)
    u, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    axis = vt[0] / np.linalg.norm(vt[0])
    proj = (pts - centroid) @ axis
    end_a, end_b = centroid + axis * proj.min(), centroid + axis * proj.max()
    return {"present": True,
            "n_voxels": best["n_voxels"], "height_mm": best["height_mm"],
            "new_in_post_frac": best["new_in_post_frac"],
            "bbox_min_SAL_vox": best["bbox_min"], "bbox_max_SAL_vox": best["bbox_max"],
            "centroid_SAL_mm": centroid.tolist(),
            "long_axis_SAL": axis.tolist(),
            "end_a_SAL_mm": end_a.tolist(), "end_b_SAL_mm": end_b.tolist(),
            "radius_mm": round(_cylinder_radius(pts, centroid, axis, proj), 2),
            "n_candidates": len(cands),
            "candidates": [{k: v for k, v in c.items() if k != "idx"} for c in cands],
            "_voxels": idx}


def orient_implant(implant, pre_field, post_field, voxel_mm, iso=geom.ISO_LEVEL):
    a = np.array(implant["end_a_SAL_mm"], float)
    b = np.array(implant["end_b_SAL_mm"], float)
    axis = (b - a) / np.linalg.norm(b - a)

    def bone_beyond(end, direction):
        offs = np.arange(1.0, 4.01, 0.25)[:, None] * direction[None, :]
        pts = (end + offs) / voxel_mm
        ok = np.all((pts >= 0) & (pts < np.array(post_field.shape) - 1), axis=1)
        if not ok.any():
            return 0.0
        v = ndi.map_coordinates(post_field, pts[ok].T, order=1)
        return float(np.mean(v > iso))

    bone_beyond_b = bone_beyond(b, axis)
    bone_beyond_a = bone_beyond(a, -axis)
    if bone_beyond_b >= bone_beyond_a:
        platform, apex = a, b
    else:
        platform, apex = b, a
    return {"platform_SAL_mm": platform.tolist(), "apex_SAL_mm": apex.tolist(),
            "bone_fraction_beyond_end_a": round(bone_beyond_a, 3),
            "bone_fraction_beyond_end_b": round(bone_beyond_b, 3),
            "decision": ""}


def find_buccal_direction(pre_field, post_field, centre_SAL_mm, axis_SAL, voxel_mm,
                          radius_mm=9.0, iso=geom.ISO_LEVEL):
    axis = np.asarray(axis_SAL, float)
    axis = axis / np.linalg.norm(axis)
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(tmp @ axis) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    t = np.arange(0.0, 10.01, 0.5)
    r = np.arange(1.0, radius_mm + 0.01, 0.5)
    th = np.arange(0.0, 2 * np.pi, np.pi / 36)
    T, R, TH = np.meshgrid(t, r, th, indexing="ij")
    dirs = (np.cos(TH)[..., None] * u + np.sin(TH)[..., None] * v)
    P = (np.asarray(centre_SAL_mm, float) + T[..., None] * axis + R[..., None] * dirs)
    idx = P.reshape(-1, 3) / voxel_mm
    ok = np.all((idx >= 0) & (idx < np.array(pre_field.shape) - 1), axis=1)
    pre_v = np.full(len(idx), -1000.0)
    post_v = np.full(len(idx), -1000.0)
    pre_v[ok] = ndi.map_coordinates(pre_field, idx[ok].T, order=1)
    post_v[ok] = ndi.map_coordinates(post_field, idx[ok].T, order=1)
    gained = ((post_v > iso) & (pre_v <= iso)).reshape(T.shape)

    weight = gained.sum(axis=(0, 1)).astype(float)
    if weight.sum() < 10:
        return None
    vec = (weight[:, None] * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)).sum(axis=0)
    nrm = np.linalg.norm(vec)
    e_buccal = vec / nrm
    cosines = (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v) @ e_buccal
    conc = float((weight * cosines).sum() / weight.sum())
    return {"e_buccal_SAL": e_buccal.tolist(),
            "concentration": round(conc, 3),
            "n_gained_samples": int(gained.sum()),
            "interpretation": ""}


def standard_roi(size_md_mm=10.0, size_ap_mm=10.0, offset=None, anchor="implant"):
    off = np.array([15.0, 15.0, 15.0] if offset is None else offset, float)
    return {"X_mm": [off[0] - size_md_mm / 2, off[0] + size_md_mm / 2],
            "Y_mm": [off[1], off[1] + size_ap_mm],
            "size_md_mm": size_md_mm, "size_ap_mm": size_ap_mm,
            "projected_area_mm2": size_md_mm * size_ap_mm,
            "anchor": anchor,
            "definition": (
                (f"{size_md_mm:.0f} mm along the ridge centred on the midpoint of the line "
                 f"connecting the buccal CEJ of the two teeth bounding the edentulous site x "
                 f"{size_ap_mm:.0f} mm apical from the PRE-OPERATIVE buccal crest")
                if anchor == "preop_anatomy" else
                (f"{size_md_mm:.0f} mm along the ridge centred on the implant long axis x "
                 f"{size_ap_mm:.0f} mm apical from the implant platform")
            ) + ("; volume integrated bucco-lingually between the pre-op and post-op bone "
                 "surfaces")}


def control_windows_clean(roi, profile, gap_mm=2.0, frac=0.20, max_reach_mm=45.0,
                          md_positive_is="mesial"):
    w = roi["size_md_mm"]
    x = np.asarray(profile["x_mm"], float)
    v = np.asarray(profile["new_bone_mm3_per_slab"], float)
    from .volume import _profile_baseline
    if v.size and v.max() > 0:
        pk = int(np.argmax(v))
        base = _profile_baseline(x, v, pk)
        thr = base + frac * (v[pk] - base)
    else:
        thr = np.inf

    cov = np.asarray(profile.get("coverage", np.ones_like(v)), float)

    def clean(lo, hi):
        m = (x >= lo) & (x <= hi)
        if not m.any():
            return False
        if float(cov[m].mean()) < 0.90:
            return False
        return bool(v[m].mean() <= thr)

    lo_name, hi_name = (("distal", "mesial") if md_positive_is == "mesial"
                        else ("mesial", "distal"))
    out, meta = {}, {}
    for name, sgn in ((lo_name, -1.0), (hi_name, +1.0)):
        edge = roi["X_mm"][0] if sgn < 0 else roi["X_mm"][1]
        found = None
        for extra in np.arange(gap_mm, max_reach_mm, 1.0):
            lo = edge - extra - w if sgn < 0 else edge + extra
            hi = lo + w
            if clean(lo, hi):
                found = (lo, hi, float(extra))
                break
        if found is None:
            lo = edge - gap_mm - w if sgn < 0 else edge + gap_mm
            out[name] = [lo, lo + w]
            meta[name] = {"clean": False, "gap_mm": gap_mm,
                          "WARNING": (f"no augmentation-free {w:.0f} mm window within "
                                      f"{max_reach_mm:.0f} mm on the {name} side; fell back to "
                                      f"the fixed {gap_mm:.0f} mm gap, so this control may be "
                                      f"contaminated by graft and the background offset "
                                      f"correspondingly inflated")}
        else:
            out[name] = [found[0], found[1]]
            meta[name] = {"clean": True, "gap_mm": round(found[2], 1)}
    out["_placement"] = meta
    out["_rule"] = (f"nearest {w:.0f} mm window either side of the ROI, at least {gap_mm:.0f} mm "
                    f"clear of it, whose MEAN new-bone slab volume does not exceed the far-field "
                    f"baseline plus {frac:.0%} of (peak - baseline), and whose mean mesiodistal "
                    f"coverage is at least 90%. The test is a mean over the window against a "
                    f"baseline-corrected threshold, not a per-slab test against a fraction of the "
                    f"raw peak; see the docstring of clean() for the measured reason.")
    return out


def control_windows(roi, gap_mm=2.0):
    w = roi["size_md_mm"]
    return {"distal": [roi["X_mm"][0] - gap_mm - w, roi["X_mm"][0] - gap_mm],
            "mesial": [roi["X_mm"][1] + gap_mm, roi["X_mm"][1] + gap_mm + w]}

