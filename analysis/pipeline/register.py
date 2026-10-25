from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from . import geom

STAGES = (
    ("coarse", 400, 1.00, [8, 4, 2], [4.0, 2.0, 1.0], 0.30),
    ("fine",   600, 0.10, [2, 1],    [1.0, 0.0],      0.60),
)


def site_box(pre_vol, post_vol, margin_vox=38, max_half_extent_vox=70):
    changed = post_vol > 700
    changed &= (pre_vol <= 700)
    changed = ndi.binary_opening(changed, np.ones((3, 3, 3)))
    if changed.sum() < 200:
        return None
    lbl, n = ndi.label(changed)
    counts = np.bincount(lbl.ravel())
    big = lbl == (1 + int(np.argmax(counts[1:])))
    idx = np.argwhere(big)
    centre = np.percentile(idx, 50, axis=0)
    lo_p = np.percentile(idx, 1, axis=0) - margin_vox
    hi_p = np.percentile(idx, 99, axis=0) + margin_vox
    half = np.minimum(np.maximum(centre - lo_p, hi_p - centre), max_half_extent_vox)
    lo = np.maximum(np.floor(centre - half), 0).astype(int)
    hi = np.minimum(np.ceil(centre + half), np.array(pre_vol.shape)).astype(int)
    return lo, hi


def _metric_mask(img, pre_vol, post_vol, lo_hi, lo=500, hi=2800):
    assert pre_vol is not None, "_metric_mask needs the calibrated pre array"
    m = pre_vol > lo
    m &= (pre_vol < hi)
    m = m.view(np.uint8)
    if lo_hi is not None:
        lo_i, hi_i = lo_hi
        m[lo_i[0]:hi_i[0], lo_i[1]:hi_i[1], lo_i[2]:hi_i[2]] = 0
    mi = sitk.GetImageFromArray(m)
    mi.CopyInformation(img)
    return mi, int(m.sum())


def control_ray_residual(pre_vol, post_vol, site_lo_hi, voxel_mm, level=geom.ISO_LEVEL,
                         n_rays=60, seed=11):
    P = geom.bone_field(pre_vol)
    Q = geom.bone_field(post_vol)
    rng = np.random.default_rng(seed)
    Z, Y, X = pre_vol.shape

    if site_lo_hi is not None:
        lo, hi = site_lo_hi
    else:
        lo = hi = None
    bone_any = P > level
    zs, ys = np.where(bone_any.any(axis=2))
    if not len(zs):
        return None
    pick = rng.choice(len(zs), size=min(n_rays * 12, len(zs)), replace=False)
    first, last = [], []
    for k in pick:
        z, y = int(zs[k]), int(ys[k])
        if lo is not None and (lo[0] <= z < hi[0]) and (lo[1] <= y < hi[1]):
            continue
        rp = np.where(P[z, y] > level)[0]
        rq = np.where(Q[z, y] > level)[0]
        if len(rp) < 3 or len(rq) < 3:
            continue
        first.append((rp[0] - rq[0]) * voxel_mm)
        last.append((rq[-1] - rp[-1]) * voxel_mm)
        if len(first) >= n_rays:
            break
    if len(first) < 8:
        return None
    f, l = float(np.median(first)), float(np.median(last))
    return {"n_rays": len(first),
            "low_edge_outward_mm": round(f, 3),
            "high_edge_outward_mm": round(l, 3),
            "median_mm": round(0.5 * (abs(f) + abs(l)) * np.sign(f + l) if (f + l) else 0.0, 3),
            "rigid_shift_mm": round((f - l) / 2.0, 3),
            "dilation_mm": round((f + l) / 2.0, 3),
            "interpretation": ""}


def _run_stages(fixed, moving, tx, mask):
    for label, iters, lr, shrink, sigmas, pct in STAGES:
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        R.SetMetricFixedMask(mask)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(pct, seed=7)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iters,
                                        convergenceMinimumValue=1e-9, convergenceWindowSize=40,
                                        estimateLearningRate=R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(shrink)
        R.SetSmoothingSigmasPerLevel(sigmas)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(tx, inPlace=False)
        tx = R.Execute(fixed, moving)
    return tx


def _stable_bone_qc(pre_vol, post_arr, lo_hi):
    m = (pre_vol > 500) & (pre_vol < 2800)
    if lo_hi is not None:
        lo, hi = lo_hi
        m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
    if m.sum() < 1000:
        return {"n_voxels": int(m.sum())}
    return {"n_voxels": int(m.sum()),
            "mean_abs_difference": round(float(np.abs(post_arr[m] - pre_vol[m]).mean()), 1),
            "median_mm": None}


def register(case, pre_vol, voxel_mm, cfg, out_path, pre_path=None, post_path=None):
    fixed = sitk.ReadImage(pre_path or f"cbct_blinded/{case}/pre.nii.gz", sitk.sitkFloat32)
    moving = sitk.ReadImage(post_path or f"cbct_blinded/{case}/post.nii.gz", sitk.sitkFloat32)

    init_tfm = cfg.get("initial_transform")
    if init_tfm:
        init = sitk.ReadTransform(init_tfm)
    else:
        init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                 sitk.CenteredTransformInitializerFilter.GEOMETRY)
    full_mask, _ = _metric_mask(fixed, pre_vol, None, None)
    tx = _run_stages(fixed, moving, init, full_mask)
    del full_mask
    pass1 = sitk.GetArrayFromImage(
        sitk.Resample(moving, fixed, tx, sitk.sitkLinear, -1000.0, sitk.sitkFloat32))

    lo_hi = cfg.get("site_box") or site_box(pre_vol, pass1)
    if lo_hi is not None and isinstance(lo_hi, list):
        lo_hi = (np.array(lo_hi[0]), np.array(lo_hi[1]))
    mask, n_vox = _metric_mask(fixed, pre_vol, pass1, lo_hi)
    bone_total = int(((pre_vol > 500) & (pre_vol < 2800)).sum())
    if n_vox < 0.25 * bone_total:
        raise SystemExit(f"ABORT: the surgical-site exclusion would leave only "
                         f"{100 * n_vox / bone_total:.0f} % of the bone in the registration "
                         f"metric. Site detection has failed; set 'site_box' in the case config.")

    qc0 = _stable_bone_qc(pre_vol, pass1, lo_hi)
    del pass1
    for label, iters, lr, shrink, sigmas, pct in STAGES:
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        R.SetMetricFixedMask(mask)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(pct, seed=7)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iters,
                                        convergenceMinimumValue=1e-9, convergenceWindowSize=40,
                                        estimateLearningRate=R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel(shrink)
        R.SetSmoothingSigmasPerLevel(sigmas)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(tx, inPlace=False)
        tx = R.Execute(fixed, moving)

    post_reg = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    sitk.WriteImage(post_reg, out_path)
    post_arr = sitk.GetArrayFromImage(post_reg)
    qc1 = _stable_bone_qc(pre_vol, post_arr, lo_hi)

    inside = post_arr > -900.0
    A = (pre_vol > 500) & inside
    B = (post_arr > 500) & inside
    inter = float((A & B).sum())
    dice = round(2.0 * inter / max(float(A.sum()) + float(B.sum()), 1.0), 3)
    bone_all = pre_vol > 500
    coverage = round(float((bone_all & inside).sum()) / max(float(bone_all.sum()), 1.0), 3)
    del A, B, inside, bone_all

    p = list(tx.GetParameters()) if hasattr(tx, "GetParameters") else []
    info = {"method": "",
            "passes": "",
            "site_box_excluded_voxels": None if lo_hi is None
            else [np.asarray(lo_hi[0]).tolist(), np.asarray(lo_hi[1]).tolist()],
            "bone_mask_dice": dice,
            "pre_bone_inside_post_fov": coverage,
            "bone_mask_dice_note": (""),
            "initial_transform": init_tfm,
            "metric_bone_fraction_kept": round(n_vox / max(bone_total, 1), 3),
            "metric_mask_voxels": n_vox,
            "parameters_rot_rad_transl_mm": [round(float(x), 5) for x in p],
            "v_initial": dict(qc0 or {}, label="initial"),
            "v_final": dict(qc1 or {}, label="after refinement"),
            "qc": ""}
    return post_arr, info

