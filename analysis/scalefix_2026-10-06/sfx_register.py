#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "analysis"))
from pipeline import geom
from pipeline.register import STAGES

SFX = ROOT / "analysis/scalefix_2026-10-06"
RULES = SFX / "PREDECLARED_RULES_2026-10-06.md"
RULES_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SITEBOX_JSON = ROOT / "FINAL_ARCH_20260907/registration_sitebox_rule_2026-09-14.json"
ERODE_MM = 0.6
FOV_CLEAR_MM = 3.0
MASK_TAG = "_v2"
N_SECTORS = 8
SEEDS = (1, 2, 3, 4, 5)
BONE_LO, BONE_HI = 500.0, 2800.0


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def write_new(p, obj):
    p = Path(p)
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def case_dir(case):
    d = SFX / case
    d.mkdir(parents=True, exist_ok=True)
    return d


def frame_A(case):
    r = json.loads((ROOT / f"analysis/{case}/results.json").read_text())
    return geom.LocalFrame.from_dict(r["frame"]), r


def build_mask(case):
    out_nii = case_dir(case) / f"maxilla_metric_mask{MASK_TAG}.nii.gz"
    out_json = case_dir(case) / f"maxilla_metric_mask{MASK_TAG}.json"
    assert sha256_file(RULES) == RULES_SHA, "declared rules changed"
    fixed = sitk.ReadImage(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"), sitk.sitkFloat32)
    pre = sitk.GetArrayFromImage(fixed)
    vx = float(fixed.GetSpacing()[0])
    bone = (pre > BONE_LO) & (pre < BONE_HI)
    pad = float(pre[0, 0, 0])
    assert np.all(pre[:, :3, :3] == pad), "corner is not a constant padding value"
    fov = ndi.binary_erosion(pre != pad, iterations=int(round(FOV_CLEAR_MM / vx)))
    bone_all = int(bone.sum())
    bone &= fov
    del fov
    er = int(round(ERODE_MM / vx))
    e = ndi.binary_erosion(bone, iterations=er)
    lab, n = ndi.label(e)
    cnt = np.bincount(lab.ravel()); cnt[0] = 0
    cfg = json.loads((ROOT / f"cases/{case}.json").read_text())
    lm = cfg["landmarks_preop_SAL_mm"]
    votes = []
    rad = int(round(2.0 / vx))
    for k in geom.ANATOMY_LANDMARKS:
        p = np.round(np.asarray(lm[k], float) / vx).astype(int)
        sl = tuple(slice(max(a - rad, 0), a + rad + 1) for a in p)
        v = lab[sl]; v = v[v > 0]
        if len(v):
            votes.extend(v.tolist())
    assert votes, "no labelled bone near any landmark"
    vc = np.bincount(np.array(votes))
    maxl = int(np.argmax(vc))
    maxi = lab == maxl
    others = (cnt >= 1000); others[maxl] = False; others[0] = False
    oth = others[lab]
    del lab, e
    keep = bone & ndi.binary_dilation(maxi, iterations=er + 1)
    keep &= ~ndi.binary_dilation(oth, iterations=er + 1)
    del maxi, oth
    sb = json.loads(SITEBOX_JSON.read_text())["cases"][case]
    (lo, hi) = [np.array(v) for v in sb["site_box_voxels_SAL_lo_hi"]]
    keep_nobox = int(keep.sum())
    keep[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
    m = keep.astype(np.uint8)
    mi = sitk.GetImageFromArray(m); mi.CopyInformation(fixed)
    if out_nii.exists():
        raise SystemExit(f"{out_nii} exists; not overwritten")
    sitk.WriteImage(mi, str(out_nii), True)
    rec = {"case": case, "rules_sha256": RULES_SHA, "voxel_mm": vx, "erosion_vox": er,
           "bone_voxels_all": bone_all, "bone_voxels_inside_fov_clear": int(bone.sum()),
           "fov_padding_value": pad, "fov_clearance_mm": FOV_CLEAR_MM, "amendment": "",
           "n_components_eroded": int(n),
           "maxilla_label_votes": {str(i): int(c) for i, c in enumerate(vc) if c},
           "other_components_ge_1000": int(others.sum()),
           "maxilla_voxels_before_sitebox": keep_nobox, "mask_voxels": int(m.sum()),
           "fraction_of_bone_kept": round(float(m.sum()) / max(bone_all, 1), 3),
           "site_box_voxels_SAL_lo_hi": [lo.tolist(), hi.tolist()],
           "site_box_source": str(SITEBOX_JSON.relative_to(ROOT)),
           "mask_file": str(out_nii.relative_to(ROOT)), "peak_rss_GiB": round(rss_gib(), 2),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rec["mask_sha256"] = sha256_file(out_nii)
    write_new(out_json, rec)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (axis, nm) in enumerate(((2, "sagittal (MIP over L)"), (1, "coronal (MIP over A)"), (0, "axial (MIP over S)"))):
        b = bone.max(axis=axis).astype(float); k = m.max(axis=axis).astype(float)
        rgb = np.dstack([0.45 * b + 0.55 * (b * (1 - k)), 0.45 * b + 0.55 * k, 0.45 * b])
        ax[i].imshow(rgb, origin="lower" if axis != 0 else "upper", aspect="equal")
        ax[i].set_title(f"{case} {nm}: green = metric mask, red = bone excluded")
    fig.tight_layout()
    fig.savefig(case_dir(case) / f"maxilla_metric_mask{MASK_TAG}_QC.png", dpi=90)
    print(json.dumps(rec, indent=1))


def load_mask(case):
    rec = json.loads((case_dir(case) / f"maxilla_metric_mask{MASK_TAG}.json").read_text())
    p = ROOT / rec["mask_file"]
    assert sha256_file(p) == rec["mask_sha256"]
    return sitk.ReadImage(str(p), sitk.sitkUInt8), rec


def init_from_rigid(case, kind):
    e = sitk.ReadTransform(str(ROOT / f"analysis/{case}/sitebox_sensitivity_2026-09-14/landmark_box/"
                                      "transform_resample2_A_leaf0.tfm"))
    e = e.Downcast()
    assert e.GetName() == "Euler3DTransform", e.GetName()
    if kind == "similarity":
        t = sitk.Similarity3DTransform()
    elif kind == "aniso":
        t = sitk.ScaleVersor3DTransform()
    elif kind == "rigid":
        t = sitk.VersorRigid3DTransform()
    else:
        raise ValueError(kind)
    mask_img, _ = load_mask(case)
    m = sitk.GetArrayFromImage(mask_img)
    ci = np.argwhere(m > 0).mean(0)[::-1]
    c_new = np.array(mask_img.TransformContinuousIndexToPhysicalPoint(ci.tolist()))
    M = np.array(e.GetMatrix()).reshape(3, 3)
    c = np.array(e.GetCenter()); tr = np.array(e.GetTranslation())
    t_new = M @ (c_new - c) + c + tr - c_new
    if kind == "aniso":
        vr = sitk.VersorRigid3DTransform(); vr.SetCenter(c_new.tolist()); vr.SetMatrix(e.GetMatrix())
        vr.SetTranslation(t_new.tolist())
        t.SetCenter(c_new.tolist())
        t.SetParameters(list(vr.GetParameters()) + [1.0, 1.0, 1.0])
    else:
        t.SetCenter(c_new.tolist())
        t.SetMatrix(e.GetMatrix())
        t.SetTranslation(t_new.tolist())
    probe = [c_new + np.array(v) for v in ((0, 0, 0), (30, 0, 0), (0, 30, 0), (0, 0, 30))]
    for q in probe:
        assert np.allclose(t.TransformPoint(q.tolist()), e.TransformPoint(q.tolist()), atol=1e-9), "re-centring changed the map"
    return t


def run_stages(fixed, moving, tx, mask, stages=STAGES, seed=7, metric="mmi", weights=None):
    hist = []
    for label, iters, lr, shrink, sigmas, pct in stages:
        R = sitk.ImageRegistrationMethod()
        if metric == "mmi":
            R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
        else:
            R.SetMetricAsCorrelation()
        R.SetMetricFixedMask(mask)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetMetricSamplingPercentage(pct, seed=seed)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=iters,
                                        convergenceMinimumValue=1e-9, convergenceWindowSize=40,
                                        estimateLearningRate=R.EachIteration)
        R.SetOptimizerScalesFromPhysicalShift()
        if weights is not None:
            R.SetOptimizerWeights(weights)
        R.SetShrinkFactorsPerLevel(shrink)
        R.SetSmoothingSigmasPerLevel(sigmas)
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(tx, inPlace=True)
        R.Execute(fixed, moving)
        hist.append({"stage": label, "metric": R.GetMetricValue(), "iterations": R.GetOptimizerIteration(),
                     "stop": R.GetOptimizerStopConditionDescription()})
    return tx, hist


def describe(tx):
    p = list(tx.GetParameters())
    out = {"type": tx.GetName(), "parameters": p, "fixed_parameters": list(tx.GetFixedParameters()),
           "matrix": list(tx.GetMatrix()), "translation": list(tx.GetTranslation()),
           "center": list(tx.GetCenter())}
    M = np.array(tx.GetMatrix()).reshape(3, 3)
    if tx.GetName() == "Similarity3DTransform":
        out["scale"] = float(p[6])
        Rm = M / p[6]
    elif tx.GetName() == "ScaleVersor3DTransform":
        out["scale_xyz_LAS"] = [float(v) for v in p[6:9]]
        out["scale_SAL"] = [float(p[8]), float(p[7]), float(p[6])]
        Rm = None
    else:
        Rm = M
    if Rm is not None:
        out["rotation_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))
    return out


def load_pair(case):
    fixed = sitk.ReadImage(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(ROOT / f"analysis/{case}/post_cal.nii.gz"), sitk.sitkFloat32)
    return fixed, moving


def qc_after(fixed, moving, tx, mask_img):
    post = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
    post_arr = sitk.GetArrayFromImage(post)
    pre = sitk.GetArrayFromImage(fixed)
    inside = post_arr > -900.0
    A = (pre > 500) & inside
    B = (post_arr > 500) & inside
    dice = round(2.0 * float((A & B).sum()) / max(float(A.sum()) + float(B.sum()), 1.0), 4)
    bone_all = pre > 500
    cov = round(float((bone_all & inside).sum()) / max(float(bone_all.sum()), 1.0), 4)
    m = sitk.GetArrayFromImage(mask_img).astype(bool)
    dice_m = round(2.0 * float((A & B & m).sum()) / max(float((A & m).sum()) + float((B & m).sum()), 1.0), 4)
    return post, {"bone_mask_dice_in_overlap": dice, "pre_bone_inside_post_fov": cov,
                  "bone_mask_dice_in_metric_mask": dice_m}


def primary(case, kind="similarity", tag=None):
    tag = tag or kind
    d = case_dir(case)
    outj = d / f"reg_{tag}.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists; not overwritten")
    t0 = time.time()
    fixed, moving = load_pair(case)
    mask, mrec = load_mask(case)
    tx = init_from_rigid(case, kind)
    init = describe(tx)
    tx, hist = run_stages(fixed, moving, tx, mask)
    tfm = d / f"reg_{tag}.tfm"
    sitk.WriteTransform(tx, str(tfm))
    post, qc = qc_after(fixed, moving, tx, mask)
    rec = {"case": case, "kind": kind, "rules_sha256": RULES_SHA, "threads": os.environ.get(
        "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"), "itk_threads_reported": sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(),
        "initial": init, "final": describe(tx), "stages": hist, "qc": qc, "mask": mrec,
        "transform_file": str(tfm.relative_to(ROOT)), "transform_sha256": sha256_file(tfm)}
    if kind in ("similarity", "aniso", "rigid"):
        out = d / f"post_reg_{tag}.nii.gz"
        if out.exists():
            raise SystemExit(f"{out} exists")
        sitk.WriteImage(post, str(out))
        rec["post_reg_file"] = str(out.relative_to(ROOT)); rec["post_reg_sha256"] = sha256_file(out)
    rec["runtime_s"] = round(time.time() - t0, 1); rec["peak_rss_GiB"] = round(rss_gib(), 2)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_new(outj, rec)
    print(json.dumps({k: rec[k] for k in ("case", "kind", "qc", "runtime_s", "peak_rss_GiB")}, indent=1))
    print(json.dumps(rec["final"].get("scale", rec["final"].get("scale_SAL")), indent=1))


def sector_masks(case, mask_img):
    m = sitk.GetArrayFromImage(mask_img).astype(bool)
    idx = np.argwhere(m)
    c = idx[:, 1:].mean(0)
    ang = np.arctan2(idx[:, 2] - c[1], idx[:, 1] - c[0])
    sec = np.floor((ang + np.pi) / (2 * np.pi) * N_SECTORS).astype(int).clip(0, N_SECTORS - 1)
    return m, idx, sec


def refit(fixed, moving, base_tfm, mask_img, kind, seed=7, metric="mmi", weights=None, scale=None, both_stages=False):
    t = sitk.ReadTransform(str(base_tfm)).Downcast()
    if kind == "similarity":
        assert t.GetName() == "Similarity3DTransform", t.GetName()
        if scale is not None:
            t.SetScale(float(scale))
    elif kind == "aniso":
        assert t.GetName() == "ScaleVersor3DTransform", t.GetName()
    stages = STAGES if both_stages else STAGES[1:]
    t, hist = run_stages(fixed, moving, t, mask_img, stages=stages, seed=seed, metric=metric, weights=weights)
    return t, hist


def uncert(case, kind="similarity"):
    d = case_dir(case)
    tag = "similarity" if kind == "similarity" else "aniso"
    outj = d / f"uncert_{tag}.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    t0 = time.time()
    fixed, moving = load_pair(case)
    mask_img, mrec = load_mask(case)
    base = d / f"reg_{tag}.tfm"
    prim = json.loads((d / f"reg_{tag}.json").read_text())["final"]
    key = "scale" if kind == "similarity" else "scale_SAL"
    m, idx, sec = sector_masks(case, mask_img)
    rec = {"case": case, "kind": kind, "rules_sha256": RULES_SHA, "primary": prim[key], "jackknife": [], "seeds": []}
    for s in range(N_SECTORS):
        mm = m.copy()
        drop = idx[sec == s]
        mm[tuple(drop.T)] = False
        mi = sitk.GetImageFromArray(mm.astype(np.uint8)); mi.CopyInformation(mask_img)
        t, hist = refit(fixed, moving, base, mi, kind)
        rec["jackknife"].append({"sector": s, "dropped_voxels": int(len(drop)), "value": describe(t)[key],
                                 "metric": hist[-1]["metric"]})
        print(case, kind, "jk", s, len(drop), describe(t)[key], flush=True)
    vals = np.array([j["value"] for j in rec["jackknife"]], float)
    n = len(vals)
    se = np.sqrt((n - 1) / n * ((vals - vals.mean(0)) ** 2).sum(0))
    rec["jackknife_se"] = se.tolist()
    p = np.array(prim[key], float)
    rec["ci95"] = [(p - 1.96 * se).tolist(), (p + 1.96 * se).tolist()]
    for sd in SEEDS:
        t, hist = refit(fixed, moving, base, mask_img, kind, seed=sd)
        rec["seeds"].append({"seed": sd, "value": describe(t)[key]})
        print(case, kind, "seed", sd, describe(t)[key], flush=True)
    sv = np.array([s["value"] for s in rec["seeds"]], float)
    rec["seed_sd"] = sv.std(0, ddof=1).tolist()
    if kind == "similarity":
        rec["init_perturbation"] = []
        for ds in (-0.01, +0.01):
            t = sitk.ReadTransform(str(base)).Downcast()
            t.SetScale(float(prim["scale"]) + ds)
            t, hist = run_stages(fixed, moving, t, mask_img)
            rec["init_perturbation"].append({"start_scale": float(prim["scale"]) + ds,
                                             "final_scale": describe(t)["scale"],
                                             "iterations": [h["iterations"] for h in hist]})
            print(case, "perturb", ds, describe(t)["scale"], flush=True)
    if kind == "similarity":
        t = init_from_rigid(case, "similarity")
        t, hist = run_stages(fixed, moving, t, mask_img, metric="corr")
        rec["alt_correlation_metric"] = describe(t)["scale"]
        fr, res = frame_A(case)
        ext = res["graft_extent_SECONDARY"]
        mfull = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / mrec["mask_file"]))).astype(bool)
        pre = sitk.GetArrayFromImage(fixed)
        lo, hi = [np.array(v) for v in mrec["site_box_voxels_SAL_lo_hi"]]
        boxreg = np.zeros_like(mfull); boxreg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
        bone = (pre > BONE_LO) & (pre < BONE_HI)
        reach = ndi.binary_dilation(mfull, iterations=mrec["erosion_vox"] + 1)
        m2 = mfull | (boxreg & bone & reach)
        ii = np.argwhere(m2)
        loc = fr.to_local(ii * mrec["voxel_mm"])
        x0, x1 = ext["x_from_mm"] - 8.0, ext["x_to_mm"] + 8.0
        ex = (loc[:, 0] > x0) & (loc[:, 0] < x1) & (loc[:, 2] > -5.0) & (loc[:, 2] < 60.0)
        m2[tuple(ii[ex].T)] = False
        mi = sitk.GetImageFromArray(m2.astype(np.uint8)); mi.CopyInformation(mask_img)
        t = init_from_rigid(case, "similarity")
        t, hist = run_stages(fixed, moving, t, mi)
        rec["alt_graft_extent_exclusion"] = {"scale": describe(t)["scale"], "excluded_local_X_mm": [x0, x1],
                                             "mask_voxels": int(m2.sum())}
        lo95, hi95 = rec["ci95"]
        hw = (hi95 - lo95) / 2
        unstable = []
        if hw > 0.003:
            unstable.append(f"jackknife 95% CI half-width {hw:.4f} > 0.003")
        for nm, v in (("correlation metric", rec["alt_correlation_metric"]),
                      ("graft-extent exclusion", rec["alt_graft_extent_exclusion"]["scale"])):
            if not (lo95 <= v <= hi95):
                unstable.append(f"{nm} fit {v:.5f} outside the CI [{lo95:.5f}, {hi95:.5f}]")
        rec["D3_unstable"] = bool(unstable); rec["D3_reasons"] = unstable
    else:
        se_ = np.array(se); v = p
        pairs = {}
        flag = False
        for i, j, nm in ((0, 1, "S-A"), (0, 2, "S-L"), (1, 2, "A-L")):
            diff = float(v[i] - v[j]); lim = float(1.96 * np.sqrt(se_[i] ** 2 + se_[j] ** 2))
            pairs[nm] = {"diff": diff, "limit_1.96_combined_se": lim, "beyond": abs(diff) > lim}
            flag |= abs(diff) > lim
        rec["D4_pairs"] = pairs
        rec["D4_anisotropic_primary"] = bool(flag)
    rec["runtime_s"] = round(time.time() - t0, 1); rec["peak_rss_GiB"] = round(rss_gib(), 2)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_new(outj, rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("jackknife", "seeds")}, indent=1))


def bounds_aniso(case):
    d = case_dir(case)
    u = json.loads((d / "uncert_aniso.json").read_text())
    assert u["D4_anisotropic_primary"], "the anisotropic bounds apply only where the anisotropic fit is primary"
    fixed, moving = load_pair(case)
    mask_img, mrec = load_mask(case)
    lo_sal, hi_sal = u["ci95"]
    for nm, sal in (("anisolo", lo_sal), ("anisohi", hi_sal)):
        outj = d / f"reg_{nm}.json"
        if outj.exists():
            raise SystemExit(f"{outj} exists")
        t0 = time.time()
        t = sitk.ReadTransform(str(d / "reg_aniso.tfm")).Downcast()
        p = list(t.GetParameters())
        p[6], p[7], p[8] = float(sal[2]), float(sal[1]), float(sal[0])
        t.SetParameters(p)
        t, hist = run_stages(fixed, moving, t, mask_img, stages=STAGES[1:], weights=[1, 1, 1, 1, 1, 1, 0, 0, 0])
        q = t.GetParameters()
        assert abs(q[6] - sal[2]) < 1e-12 and abs(q[7] - sal[1]) < 1e-12 and abs(q[8] - sal[0]) < 1e-12
        tfm = d / f"reg_{nm}.tfm"; sitk.WriteTransform(t, str(tfm))
        post, qc = qc_after(fixed, moving, t, mask_img)
        out = d / f"post_reg_{nm}.nii.gz"
        sitk.WriteImage(post, str(out))
        rec = {"case": case, "kind": nm, "scale_SAL_fixed_at": sal, "final": describe(t), "stages": hist, "qc": qc,
               "transform_file": str(tfm.relative_to(ROOT)), "post_reg_file": str(out.relative_to(ROOT)),
               "post_reg_sha256": sha256_file(out), "runtime_s": round(time.time() - t0, 1),
               "peak_rss_GiB": round(rss_gib(), 2), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_new(outj, rec)
        print(nm, sal, qc, flush=True)


def metric_at(fixed, moving, mask, tx, seed=7, pct=0.60):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    R.SetMetricFixedMask(mask)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(pct, seed=seed)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetInitialTransform(tx)
    return float(R.MetricEvaluate(fixed, moving))


def check_aniso(case):
    d = case_dir(case)
    outj = d / "check_aniso.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    fixed, moving = load_pair(case)
    mask_img, _ = load_mask(case)
    ts = sitk.ReadTransform(str(d / "reg_similarity.tfm")).Downcast()
    ta = sitk.ReadTransform(str(d / "reg_aniso.tfm")).Downcast()
    rec = {"case": case, "metric_similarity": {}, "metric_aniso_primary": {}}
    for sd in (7, 1, 2, 3):
        rec["metric_similarity"][str(sd)] = metric_at(fixed, moving, mask_img, ts, sd)
        rec["metric_aniso_primary"][str(sd)] = metric_at(fixed, moving, mask_img, ta, sd)
    s = ts.GetScale()
    vr = sitk.VersorRigid3DTransform(); vr.SetCenter(ts.GetCenter())
    vr.SetMatrix(tuple((np.array(ts.GetMatrix()) / s).tolist())); vr.SetTranslation(ts.GetTranslation())
    t = sitk.ScaleVersor3DTransform(); t.SetCenter(ts.GetCenter())
    t.SetParameters(list(vr.GetParameters()) + [s, s, s])
    for q in ((0.0, 0.0, 0.0), (30.0, 20.0, 10.0)):
        assert np.allclose(t.TransformPoint(q), ts.TransformPoint(q), atol=1e-6)
    rec["metric_aniso_start_at_similarity"] = metric_at(fixed, moving, mask_img, t, 7)
    t, hist = run_stages(fixed, moving, t, mask_img)
    rec["aniso_from_similarity"] = {"scale_SAL": describe(t)["scale_SAL"], "stages": hist,
                                    "metric": {str(sd): metric_at(fixed, moving, mask_img, t, sd) for sd in (7, 1, 2, 3)}}
    tfm = d / "reg_aniso_from_similarity.tfm"; sitk.WriteTransform(t, str(tfm))
    rec["transform_file"] = str(tfm.relative_to(ROOT))
    ms = np.mean([rec["metric_similarity"][k] for k in rec["metric_similarity"]])
    ma = np.mean([rec["metric_aniso_primary"][k] for k in rec["metric_aniso_primary"]])
    mf = np.mean([rec["aniso_from_similarity"]["metric"][k] for k in rec["aniso_from_similarity"]["metric"]])
    rec["mean_metric"] = {"similarity": ms, "aniso_primary": ma, "aniso_from_similarity": mf}
    rec["aniso_primary_worse_than_similarity"] = bool(ma > ms)
    rec["peak_rss_GiB"] = round(rss_gib(), 2)
    write_new(outj, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "aniso_from_similarity"}, indent=1))
    print(rec["aniso_from_similarity"]["scale_SAL"], rec["aniso_from_similarity"]["metric"])


def polar(M):
    M = np.asarray(M, float).reshape(3, 3)
    w, U = np.linalg.eigh(M.T @ M)
    P = U @ np.diag(np.sqrt(w)) @ U.T
    return M @ np.linalg.inv(P), P


def affine_desc(t):
    R_, P = polar(t.GetMatrix())
    return {"P_diag_SAL": [float(P[2, 2]), float(P[1, 1]), float(P[0, 0])],
            "P_offdiag_SA_SL_AL": [float(P[2, 1]), float(P[2, 0]), float(P[1, 0])],
            "principal_stretches": sorted(np.linalg.eigvalsh(P).tolist()),
            "det": float(np.linalg.det(np.asarray(t.GetMatrix()).reshape(3, 3))),
            "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R_) - 1) / 2, -1, 1)))),
            "matrix": list(t.GetMatrix()), "translation": list(t.GetTranslation()), "center": list(t.GetCenter())}


def affine_fit(case):
    d = case_dir(case)
    outj = d / "reg_affine.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    t0 = time.time()
    fixed, moving = load_pair(case)
    mask_img, mrec = load_mask(case)
    ts = sitk.ReadTransform(str(d / "reg_similarity.tfm")).Downcast()
    t = sitk.AffineTransform(3)
    t.SetCenter(ts.GetCenter()); t.SetMatrix(ts.GetMatrix()); t.SetTranslation(ts.GetTranslation())
    for q in ((0.0, 0.0, 0.0), (30.0, 20.0, 10.0)):
        assert np.allclose(t.TransformPoint(q), ts.TransformPoint(q), atol=1e-9)
    t, hist = run_stages(fixed, moving, t, mask_img)
    tfm = d / "reg_affine.tfm"; sitk.WriteTransform(t, str(tfm))
    m_aff = {str(sd): metric_at(fixed, moving, mask_img, t, sd) for sd in (7, 1, 2, 3)}
    m_sim = {str(sd): metric_at(fixed, moving, mask_img, ts, sd) for sd in (7, 1, 2, 3)}
    better = all(m_aff[k] <= m_sim[k] for k in m_aff)
    rec = {"case": case, "model": "", "start": "the similarity solution",
           "final": affine_desc(t), "stages": hist, "metric_affine": m_aff, "metric_similarity": m_sim,
           "precondition_affine_metric_not_worse": better, "transform_file": str(tfm.relative_to(ROOT)),
           "runtime_s": round(time.time() - t0, 1), "peak_rss_GiB": round(rss_gib(), 2),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outj, rec)
    print(json.dumps({k: rec[k] for k in ("final", "metric_affine", "metric_similarity", "precondition_affine_metric_not_worse")}, indent=1))


def uncert_affine(case):
    d = case_dir(case)
    outj = d / "uncert_affine.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    fixed, moving = load_pair(case)
    mask_img, mrec = load_mask(case)
    base = d / "reg_affine.tfm"
    prim = json.loads((d / "reg_affine.json").read_text())
    if not prim["precondition_affine_metric_not_worse"]:
        write_new(outj, {"case": case, "skipped": "", "D4a_anisotropic_primary": False})
        print(case, "precondition failed; jackknife skipped")
        return
    m, idx, sec = sector_masks(case, mask_img)
    vals = []
    jk = []
    for s_ in range(N_SECTORS):
        mm = m.copy(); drop = idx[sec == s_]; mm[tuple(drop.T)] = False
        mi = sitk.GetImageFromArray(mm.astype(np.uint8)); mi.CopyInformation(mask_img)
        t = sitk.ReadTransform(str(base)).Downcast()
        t, hist = run_stages(fixed, moving, t, mi, stages=STAGES[1:])
        dsc = affine_desc(t)
        vals.append(dsc["P_diag_SAL"]); jk.append({"sector": s_, "P_diag_SAL": dsc["P_diag_SAL"]})
        print(case, "affine jk", s_, [round(v, 5) for v in dsc["P_diag_SAL"]], flush=True)
    v = np.array(vals); n = len(v)
    se = np.sqrt((n - 1) / n * ((v - v.mean(0)) ** 2).sum(0))
    p = np.array(prim["final"]["P_diag_SAL"])
    pairs, flag = {}, False
    for i, j, nm in ((0, 1, "S-A"), (0, 2, "S-L"), (1, 2, "A-L")):
        diff = float(p[i] - p[j]); lim = float(1.96 * np.sqrt(se[i] ** 2 + se[j] ** 2))
        pairs[nm] = {"diff": diff, "limit_1.96_combined_se": lim, "beyond": abs(diff) > lim}
        flag |= abs(diff) > lim
    sel = bool(flag and prim["precondition_affine_metric_not_worse"])
    rec = {"case": case, "primary_P_diag_SAL": p.tolist(), "jackknife": jk, "jackknife_se": se.tolist(),
           "ci95": [(p - 1.96 * se).tolist(), (p + 1.96 * se).tolist()], "D4_pairs": pairs,
           "precondition_affine_metric_not_worse": prim["precondition_affine_metric_not_worse"],
           "D4a_anisotropic_primary": sel, "peak_rss_GiB": round(rss_gib(), 2),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outj, rec)
    print(json.dumps({k: rec[k] for k in ("primary_P_diag_SAL", "jackknife_se", "D4_pairs", "D4a_anisotropic_primary")}, indent=1))


def profile(case):
    d = case_dir(case)
    outj = d / "scale_profile.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    fixed, moving = load_pair(case)
    mask_img, _ = load_mask(case)
    s0 = float(sitk.ReadTransform(str(d / "reg_similarity.tfm")).Downcast().GetScale())
    rows = []
    for ds in (-0.010, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.010):
        t, hist = refit(fixed, moving, d / "reg_similarity.tfm", mask_img, "similarity", scale=s0 + ds,
                        weights=[1, 1, 1, 1, 1, 1, 0])
        m = [metric_at(fixed, moving, mask_img, t, sd) for sd in (7, 1, 2, 3)]
        rows.append({"scale": s0 + ds, "metric_mean": float(np.mean(m)), "metric_by_seed": m})
        print(case, "profile", round(s0 + ds, 5), round(float(np.mean(m)), 6), flush=True)
    x = np.array([r_["scale"] for r_ in rows]); y = np.array([r_["metric_mean"] for r_ in rows])
    c2, c1, c0 = np.polyfit(x - s0, y, 2)
    vertex = s0 - c1 / (2 * c2) if c2 > 0 else None
    rec = {"case": case, "fitted_scale": s0, "rows": rows, "quadratic_vertex_scale": vertex,
           "quadratic_curvature": float(c2), "argmin_grid_scale": float(x[int(np.argmin(y))]),
           "peak_rss_GiB": round(rss_gib(), 2), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outj, rec)
    print(json.dumps({k: rec[k] for k in ("fitted_scale", "quadratic_vertex_scale", "argmin_grid_scale")}, indent=1))


def materialize(case, name):
    d = case_dir(case)
    out = d / f"post_reg_{name}.nii.gz"
    if out.exists():
        raise SystemExit(f"{out} exists")
    fixed, moving = load_pair(case)
    mask_img, _ = load_mask(case)
    t = sitk.ReadTransform(str(d / f"reg_{name}.tfm")).Downcast()
    post, qc = qc_after(fixed, moving, t, mask_img)
    sitk.WriteImage(post, str(out))
    rec = {"case": case, "name": name, "qc": qc, "post_reg_file": str(out.relative_to(ROOT)),
           "post_reg_sha256": sha256_file(out), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(d / f"materialize_{name}.json", rec)
    print(rec)


def bounds_affine(case):
    d = case_dir(case)
    u = json.loads((d / "uncert_affine.json").read_text())
    assert u["D4a_anisotropic_primary"]
    t0_ = sitk.ReadTransform(str(d / "reg_affine.tfm")).Downcast()
    R_, P = polar(t0_.GetMatrix())
    for nm, sal in (("affinelo", u["ci95"][0]), ("affinehi", u["ci95"][1])):
        P2 = P.copy()
        P2[2, 2], P2[1, 1], P2[0, 0] = sal[0], sal[1], sal[2]
        t = sitk.AffineTransform(3)
        t.SetCenter(t0_.GetCenter()); t.SetTranslation(t0_.GetTranslation()); t.SetMatrix((R_ @ P2).ravel().tolist())
        sitk.WriteTransform(t, str(d / f"reg_{nm}.tfm"))
        write_new(d / f"reg_{nm}.json", {"case": case, "kind": nm, "P_diag_SAL_fixed_at": sal, "final": affine_desc(t),
                                          "note": ""})
        materialize(case, nm)


def fixscale(case, value, name):
    d = case_dir(case)
    outj = d / f"reg_{name}.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    fixed, moving = load_pair(case)
    mask_img, _ = load_mask(case)
    t, hist = refit(fixed, moving, d / "reg_similarity.tfm", mask_img, "similarity", scale=value,
                    weights=[1, 1, 1, 1, 1, 1, 0])
    tfm = d / f"reg_{name}.tfm"; sitk.WriteTransform(t, str(tfm))
    post, qc = qc_after(fixed, moving, t, mask_img)
    out = d / f"post_reg_{name}.nii.gz"; sitk.WriteImage(post, str(out))
    write_new(outj, {"case": case, "kind": name, "scale_fixed_at": value, "final": describe(t), "stages": hist, "qc": qc,
                     "post_reg_file": str(out.relative_to(ROOT)), "post_reg_sha256": sha256_file(out)})
    print(name, value, qc)


def bounds(case):
    d = case_dir(case)
    u = json.loads((d / "uncert_similarity.json").read_text())
    fixed, moving = load_pair(case)
    mask_img, mrec = load_mask(case)
    for nm, s in (("scalelo", u["ci95"][0]), ("scalehi", u["ci95"][1])):
        outj = d / f"reg_{nm}.json"
        if outj.exists():
            raise SystemExit(f"{outj} exists")
        t0 = time.time()
        t, hist = refit(fixed, moving, d / "reg_similarity.tfm", mask_img, "similarity", scale=s,
                        weights=[1, 1, 1, 1, 1, 1, 0])
        assert abs(t.GetScale() - s) < 1e-12, "scale was not frozen"
        tfm = d / f"reg_{nm}.tfm"; sitk.WriteTransform(t, str(tfm))
        post, qc = qc_after(fixed, moving, t, mask_img)
        out = d / f"post_reg_{nm}.nii.gz"
        sitk.WriteImage(post, str(out))
        rec = {"case": case, "kind": nm, "scale_fixed_at": s, "final": describe(t), "stages": hist, "qc": qc,
               "transform_file": str(tfm.relative_to(ROOT)), "post_reg_file": str(out.relative_to(ROOT)),
               "post_reg_sha256": sha256_file(out), "runtime_s": round(time.time() - t0, 1),
               "peak_rss_GiB": round(rss_gib(), 2), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        write_new(outj, rec)
        print(nm, s, qc, flush=True)


if __name__ == "__main__":
    case, mode = sys.argv[1], sys.argv[2]
    assert case in ("S1", "S2", "S3", "S4")
    if mode == "mask":
        build_mask(case)
    elif mode == "primary":
        assert os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS") == "1"
        primary(case, "similarity")
    elif mode == "rigid":
        primary(case, "rigid")
    elif mode == "aniso":
        primary(case, "aniso")
    elif mode == "uncert":
        uncert(case, "similarity")
    elif mode == "uncert_aniso":
        uncert(case, "aniso")
    elif mode == "bounds":
        bounds(case)
    elif mode == "bounds_aniso":
        bounds_aniso(case)
    elif mode == "check_aniso":
        check_aniso(case)
    elif mode == "affine":
        affine_fit(case)
    elif mode == "profile":
        profile(case)
    elif mode == "fixscale":
        fixscale(case, float(sys.argv[3]), sys.argv[4])
    elif mode == "materialize":
        materialize(case, sys.argv[3])
    elif mode == "bounds_affine":
        bounds_affine(case)
    elif mode == "uncert_affine":
        uncert_affine(case)
    else:
        raise SystemExit(mode)

