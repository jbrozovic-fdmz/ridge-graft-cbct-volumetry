#!/usr/bin/env python3
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import geom

TAG = "reverify_2026-09-13"
CROP_MM = 40.0
T_PRIMARY = 1500.0
T_SENS = (1000.0, 2000.0)
RAY_STEP = 0.05
RAY_HALF = 7.0
GAP_FILL_MM = 1.2
CROWN_EXT_MM = 8.0
APEX_STOP_MM = 2.0
X_OFFSETS = np.arange(-1.5, 1.51, 0.5)
CYL_R_MM = 4.5
MASK_DIL_MM = 1.0
PIN_MARGIN_MM = 2.0
VALIDATION_SHIFT_MM = 0.30
N_BOOT = 2000
SEED = 20260913
FROZEN = Path("FINAL_ARCH_20260907/FROZEN_RESULTS.json")
OUT_FINAL = Path(f"FINAL_ARCH_20260907/site_registration_{TAG}.json")

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def r(x, n=3):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def header(path):
    rd = sitk.ImageFileReader()
    rd.SetFileName(str(path))
    rd.ReadImageInformation()
    sp = np.array(rd.GetSpacing(), float)
    assert np.allclose(sp, sp[0], atol=1e-6)
    return float(sp[0]), tuple(int(s) for s in rd.GetSize()[::-1])


def read_crop(path, sl):
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayViewFromImage(img)
    c = a[sl].astype(np.float32)
    del a, img
    return c


class Site:
    def __init__(self, case):
        self.case = case
        self.res = json.loads(Path(f"analysis/{case}/results.json").read_text())
        self.frame = geom.LocalFrame.from_dict(self.res["frame"])
        self.lm = {k: np.array(v, float) for k, v in
                   self.res["anchor"]["landmarks_local_XYZ_mm"].items()}
        self.pre_path = Path(f"analysis/{case}/pre_cal.nii.gz")
        self.post_path = Path(f"cbct_blinded/{case}/post_reg.nii.gz")
        self.vx, self.shape = header(self.pre_path)
        vq, shq = header(self.post_path)
        assert abs(vq - self.vx) < 1e-6 and shq == self.shape, "post_reg not on the pre grid"
        pins = json.loads(Path(f"analysis/{case}/fixation_pins_{'2026-09-13'}.json").read_text())
        self.pins = [{"label": c["label"], "local_XYZ_mm": c["local_A"]["centroid_XYZ_mm"],
                      "length_mm": c["long_axis_length_mm"]}
                     for c in pins["components"] if c.get("pin_candidate")]

    def crop(self):
        pts = []
        for t in ("mesial", "distal"):
            cej, ap = self.lm[f"{t}_tooth_cej_buccal"], self.lm[f"{t}_tooth_root_apex"]
            pts += [cej, ap, cej + np.array([0.0, -CROWN_EXT_MM, 0.0])]
        c = self.frame.to_anatomical(np.array(pts)).mean(axis=0) / self.vx
        n = int(round(CROP_MM / self.vx))
        lo = np.maximum(np.round(c - n / 2).astype(int), 0)
        hi = np.minimum(lo + n, np.array(self.shape))
        assert np.all((hi - lo) * self.vx <= CROP_MM + 1e-6)
        self.lo = lo
        self.sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        return self.sl

    def idx_of_local(self, P):
        return self.frame.to_anatomical(P) / self.vx - self.lo

    def tooth_axes(self, pre_s):
        axes = {}
        for t in ("mesial", "distal"):
            cej, ap = self.lm[f"{t}_tooth_cej_buccal"], self.lm[f"{t}_tooth_root_apex"]
            z = np.arange(cej[2] - 9.0, cej[2] + 1.0 + 1e-9, RAY_STEP)
            loc = np.c_[np.full_like(z, cej[0]), np.full_like(z, cej[1]), z]
            v = ndi.map_coordinates(pre_s, self.idx_of_local(loc).T, order=1, cval=np.nan)
            run = ndi.binary_closing(np.nan_to_num(v, nan=-1e9) > T_PRIMARY,
                                     np.ones(int(round(GAP_FILL_MM / RAY_STEP))))
            k0 = int(np.argmin(np.abs(z - (cej[2] - 1.5))))
            lab, _ = ndi.label(run)
            if lab[k0]:
                ks = np.where(lab == lab[k0])[0]
                zc, src = 0.5 * (z[ks[0]] + z[ks[-1]]), "data: midpoint of pre dentine run at CEJ level"
                width = z[ks[-1]] - z[ks[0]]
            else:
                zc, src, width = cej[2] - 3.5, "fallback: CEJ - 3.5 mm", None
            axes[t] = {"cej_centre": np.array([cej[0], cej[1], zc]), "apex": ap,
                       "source": src, "cej_run_width_mm": r(width)}
        return axes

    @staticmethod
    def axis_at(ax, Y):
        c, a = ax["cej_centre"], ax["apex"]
        t = (np.asarray(Y, float) - c[1]) / (a[1] - c[1])
        return c[0] + t * (a[0] - c[0]), c[2] + t * (a[2] - c[2])


def edges_on_ray(v, z, T):
    if np.isnan(v).all():
        return None
    above = np.nan_to_num(v, nan=-1e9) > T
    above = ndi.binary_closing(above, np.ones(int(round(GAP_FILL_MM / RAY_STEP))))
    n = len(v)
    k0 = n // 2
    lab, _ = ndi.label(above)
    if not lab[k0]:
        w = int(round(1.5 / RAY_STEP))
        cand = [k for k in range(max(0, k0 - w), min(n, k0 + w + 1)) if lab[k]]
        if not cand:
            return None
        k0 = min(cand, key=lambda k: abs(k - n // 2))
    ks = np.where(lab == lab[k0])[0]
    a, b = int(ks[0]), int(ks[-1])
    if a == 0 or b == n - 1:
        return None
    seg = v[max(0, a - int(1.4 / RAY_STEP)): min(n, b + int(1.4 / RAY_STEP) + 1)]
    if np.isnan(seg).any():
        return None
    while a < b and not v[a] > T:
        a += 1
    while b > a and not v[b] > T:
        b -= 1
    if v[a - 1] >= T or v[b + 1] >= T:
        return None
    zp = z[a - 1] + (T - v[a - 1]) / (v[a] - v[a - 1]) * RAY_STEP
    zb = z[b] + (v[b] - T) / (v[b] - v[b + 1]) * RAY_STEP
    o1, o2 = int(0.4 / RAY_STEP), int(1.2 / RAY_STEP)
    ext_b = float(np.median(v[b + o1: b + o2 + 1]))
    ext_p = float(np.median(v[max(0, a - o2): a - o1 + 1]))
    return zp, zb, ext_p, ext_b


def pin_hit(site, X, Y, z0, z1):
    for p in site.pins:
        c = np.array(p["local_XYZ_mm"])
        zc = min(max(c[2], z0), z1)
        if np.linalg.norm(np.array([X, Y, zc]) - c) < 0.5 * p["length_mm"] + PIN_MARGIN_MM:
            return True
    return False


def method_B(site, axes, pre_s, post_s, T, shift_local=None):
    rows = []
    z_rel = np.arange(-RAY_HALF, RAY_HALF + 1e-9, RAY_STEP)
    for t in ("mesial", "distal"):
        ax = axes[t]
        ycej, yap = ax["cej_centre"][1], ax["apex"][1]
        for Y in np.arange(ycej - CROWN_EXT_MM, yap - APEX_STOP_MM + 1e-9, 0.5):
            if ycej - 0.5 < Y < ycej + 1.0:
                continue
            cls = "crown" if Y <= ycej - 0.5 else "root"
            xa, za = site.axis_at(ax, Y)
            for dx in X_OFFSETS:
                X = xa + dx
                loc = np.c_[np.full_like(z_rel, X), np.full_like(z_rel, Y), za + z_rel]
                vp = ndi.map_coordinates(pre_s, site.idx_of_local(loc).T, order=1, cval=np.nan)
                locq = loc if shift_local is None else loc - shift_local
                vq = ndi.map_coordinates(post_s, site.idx_of_local(locq).T, order=1, cval=np.nan)
                ep, eq = edges_on_ray(vp, loc[:, 2], T), edges_on_ray(vq, loc[:, 2], T)
                if ep is None or eq is None:
                    continue
                zp0, zb0, xp0, xb0 = ep
                zp1, zb1, xp1, xb1 = eq
                w0, w1 = zb0 - zp0, zb1 - zp1
                if not (2.5 <= w0 <= 12.0 and 2.5 <= w1 <= 12.0):
                    continue
                if cls == "crown":
                    if max(xp0, xb0, xp1, xb1) >= 800.0:
                        continue
                else:
                    if max(xp0, xb0, xp1, xb1) >= T - 500.0:
                        continue
                if pin_hit(site, X, Y, min(zp0, zp1) - 2, max(zb0, zb1) + 2):
                    continue
                b = zb1 - zb0
                l = zp0 - zp1
                rows.append((t, cls, float(X), float(Y), (b - l) / 2.0, (b + l) / 2.0, b, l))
    return rows


def summarise_B(rows, roi_centre, rng_seed=SEED, n_boot=N_BOOT):
    out = {"n_rays_total": len(rows)}
    if not rows:
        return out
    tooth = np.array([q[0] for q in rows]); cls = np.array([q[1] for q in rows])
    X = np.array([q[2] for q in rows]); Y = np.array([q[3] for q in rows])
    S = np.array([q[4] for q in rows]); D = np.array([q[5] for q in rows])
    Bo = np.array([q[6] for q in rows]); L = np.array([q[7] for q in rows])
    per = {}
    for t in ("mesial", "distal"):
        for c in ("crown", "root"):
            m = (tooth == t) & (cls == c)
            if m.sum() == 0:
                per[f"{t}_{c}"] = {"n_rays": 0}
                continue
            per[f"{t}_{c}"] = {"n_rays": int(m.sum()),
                               "n_Y_stations": int(len(np.unique(Y[m]))),
                               "X_mean_mm": r(X[m].mean(), 2), "Y_range_mm": [r(Y[m].min(), 2), r(Y[m].max(), 2)],
                               "shift_median_mm": r(np.median(S[m])),
                               "shift_IQR_mm": [r(np.percentile(S[m], 25)), r(np.percentile(S[m], 75))],
                               "dilation_median_mm": r(np.median(D[m])),
                               "buccal_outward_median_mm": r(np.median(Bo[m])),
                               "palatal_outward_median_mm": r(np.median(L[m]))}
    out["per_tooth"] = per

    def fits(mask):
        if mask.sum() < 10 or len(np.unique(tooth[mask])) < 2:
            return None
        x, y, s = X[mask], Y[mask], S[mask]
        clusters = np.array([f"{a}|{b}" for a, b in zip(tooth[mask], y)])
        uc = np.unique(clusters)
        idx_of = {u: np.where(clusters == u)[0] for u in uc}

        def plane(ii):
            A = np.c_[np.ones(len(ii)), x[ii], y[ii]]
            coef, *_ = np.linalg.lstsq(A, s[ii], rcond=None)
            return coef

        def xinterp(ii):
            tt = tooth[mask][ii]
            if len(np.unique(tt)) < 2:
                return np.nan
            mm, dd = tt == "mesial", tt == "distal"
            xm, xd = x[ii][mm].mean(), x[ii][dd].mean()
            sm, sd = np.median(s[ii][mm]), np.median(s[ii][dd])
            return sm + (roi_centre[0] - xm) / (xd - xm) * (sd - sm)

        full = np.arange(len(s))
        coef = plane(full)
        est = coef[0] + coef[1] * roi_centre[0] + coef[2] * roi_centre[1]
        xi = xinterp(full)
        rng = np.random.default_rng(rng_seed)
        bp, bx = [], []
        for _ in range(n_boot):
            pick = rng.choice(uc, size=len(uc), replace=True)
            ii = np.concatenate([idx_of[u] for u in pick])
            if len(np.unique(tooth[mask][ii])) < 2:
                continue
            c2 = plane(ii)
            bp.append(c2[0] + c2[1] * roi_centre[0] + c2[2] * roi_centre[1])
            bx.append(xinterp(ii))
        bp, bx = np.array(bp), np.array(bx)
        return {"n_rays": int(mask.sum()), "n_clusters": int(len(uc)),
                "plane_coef_a0_aX_aY": [r(c, 5) for c in coef],
                "plane_at_roi_centre_mm": r(est),
                "plane_at_roi_centre_CI95_mm": [r(np.nanpercentile(bp, 2.5)), r(np.nanpercentile(bp, 97.5))],
                "X_interp_of_tooth_medians_mm": r(xi),
                "X_interp_CI95_mm": [r(np.nanpercentile(bx, 2.5)), r(np.nanpercentile(bx, 97.5))],
                "median_shift_all_rays_mm": r(np.median(s)),
                "median_dilation_all_rays_mm": r(np.median(D[mask])),
                "Y_extrapolation_note": f"rays span Y {r(y.min(), 1)}..{r(y.max(), 1)}; ROI centre Y {roi_centre[1]}"}

    out["crown"] = fits(cls == "crown")
    out["crown_plus_root"] = fits(np.ones(len(S), bool))
    return out


def local_coords_of_crop(site, shape):
    out = np.empty(shape + (3,), np.float32)
    ja, jl = np.meshgrid(np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    for s in range(shape[0]):
        P = np.c_[np.full(ja.size, s + site.lo[0]), (ja + site.lo[1]).ravel(),
                  (jl + site.lo[2]).ravel()].astype(float) * site.vx
        out[s] = site.frame.to_local(P).reshape(shape[1], shape[2], 3).astype(np.float32)
    return out


def ball(r_vox):
    k = int(np.ceil(r_vox))
    g = np.mgrid[-k:k + 1, -k:k + 1, -k:k + 1]
    return (g ** 2).sum(0) <= r_vox ** 2 + 1e-9


def build_masks(site, axes, pre, post, pre_s, post_s):
    loc = local_coords_of_crop(site, pre.shape)
    teeth = {}
    for t in ("mesial", "distal"):
        ax = axes[t]
        Y = loc[..., 1]
        xa, za = site.axis_at(ax, Y)
        inY = (Y >= ax["cej_centre"][1] - CROWN_EXT_MM - 1.0) & (Y <= ax["apex"][1] + 1.0)
        cyl = inY & ((loc[..., 0] - xa) ** 2 + (loc[..., 2] - za) ** 2 <= CYL_R_MM ** 2)
        teeth[t] = cyl & (pre_s > T_PRIMARY)
        del xa, za, inY, cyl
    bl = ball(MASK_DIL_MM / site.vx)
    gained = ndi.binary_opening((post_s > 900) & (pre_s < 450), np.ones((3, 3, 3)))
    lost = ndi.binary_opening((pre_s > 900) & (post_s < 450), np.ones((3, 3, 3)))
    changed = ndi.binary_dilation(gained | lost, bl)
    pinmask = np.zeros(pre.shape, bool)
    for p in site.pins:
        c = np.array(p["local_XYZ_mm"], np.float32)
        pinmask |= ((loc - c) ** 2).sum(-1) <= (0.5 * p["length_mm"] + PIN_MARGIN_MM) ** 2
    pad = ndi.binary_dilation(post <= -1499.0, bl)
    border = np.zeros(pre.shape, bool)
    b = int(np.ceil(MASK_DIL_MM / site.vx)) + 1
    border[b:-b, b:-b, b:-b] = True
    excl = changed | pinmask | pad | ~border
    masks, info = {}, {}
    for t in ("mesial", "distal"):
        d = ndi.binary_dilation(teeth[t], bl)
        masks[t] = d & ~excl
        info[t] = {"tooth_core_voxels": int(teeth[t].sum()), "dilated_voxels": int(d.sum()),
                   "after_exclusion_voxels": int(masks[t].sum()),
                   "excluded_fraction": r(1 - masks[t].sum() / max(d.sum(), 1)),
                   "excluded_by_change": int((d & changed).sum()),
                   "excluded_by_pins": int((d & pinmask).sum()),
                   "excluded_by_padding_or_border": int((d & (pad | ~border)).sum())}
    masks["joint"] = masks["mesial"] | masks["distal"]
    info["joint_voxels"] = int(masks["joint"].sum())
    del loc
    return masks, info


def to_img(a, site):
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((site.vx,) * 3)
    img.SetOrigin(tuple(float(v) for v in (site.lo[::-1] * site.vx)))
    return img


def local_vec_to_LAS(site, v):
    return (np.asarray(v, float) @ site.frame.R)[::-1]


def register(site, fixed, moving, mask_arr, centre_local, init_local_shift=(0.0, 0.0, 0.0)):
    mask = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    mask.CopyInformation(fixed)
    tx = sitk.Euler3DTransform()
    c_SAL = site.frame.to_anatomical(np.atleast_2d(centre_local))[0]
    tx.SetCenter(tuple(float(v) for v in c_SAL[::-1]))
    tx.SetTranslation(tuple(float(v) for v in local_vec_to_LAS(site, init_local_shift)))
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricFixedMask(mask)
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(learningRate=0.25, minStep=1e-4,
                                               numberOfIterations=400, relaxationFactor=0.5,
                                               gradientMagnitudeTolerance=1e-8)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([2, 1])
    R.SetSmoothingSigmasPerLevel([0.4, 0.0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(tx, inPlace=False)
    ident = sitk.Euler3DTransform(); ident.SetCenter(tx.GetCenter())
    R0 = sitk.ImageRegistrationMethod()
    R0.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R0.SetMetricFixedMask(mask); R0.SetMetricSamplingStrategy(R0.NONE)
    R0.SetInterpolator(sitk.sitkLinear); R0.SetInitialTransform(ident)
    m_identity = R0.MetricEvaluate(fixed, moving)
    out = R.Execute(fixed, moving)
    return out, {"metric_at_identity": r(m_identity, 5), "metric_final": r(R.GetMetricValue(), 5),
                 "iterations_last_level": int(R.GetOptimizerIteration()),
                 "stop": R.GetOptimizerStopConditionDescription()[:160],
                 "init_local_shift_mm": list(init_local_shift)}


def displacement_local(site, tx, P_local):
    P_SAL = site.frame.to_anatomical(np.atleast_2d(P_local))
    q = np.array([tx.TransformPoint(tuple(float(v) for v in p[::-1])) for p in P_SAL])
    u_SAL = (q - P_SAL[:, ::-1])[:, ::-1]
    return u_SAL @ site.frame.R.T


def roi_grid(site, Z):
    roi = site.res["roi"]
    xs = np.arange(roi["X_mm"][0], roi["X_mm"][1] + 1e-9, 0.5)
    ys = np.arange(roi["Y_mm"][0], roi["Y_mm"][1] + 1e-9, 0.5)
    g = np.array([[x, y, Z] for x in xs for y in ys])
    return g


def describe_tx(site, tx, axes):
    d = {}
    for Z in (12.0, 15.0, 18.0):
        u = displacement_local(site, tx, roi_grid(site, Z))
        d[f"roi_Z{int(Z)}"] = {"buccal_mean_mm": r(u[:, 2].mean()), "buccal_min_mm": r(u[:, 2].min()),
                               "buccal_max_mm": r(u[:, 2].max()),
                               "md_mean_mm": r(u[:, 0].mean()), "apical_mean_mm": r(u[:, 1].mean())}
    pts = {}
    for t in ("mesial", "distal"):
        ax = axes[t]
        yc = ax["cej_centre"][1] - 4.0
        xa, za = site.axis_at(ax, yc)
        pts[f"{t}_crown_centre"] = [xa, yc, za]
        xa, za = site.axis_at(ax, 20.0)
        pts[f"{t}_axis_Y20_Z15"] = [xa, 20.0, 15.0]
    for nm, xr in site.res["control_windows"].items():
        pts[f"control_{nm}_centre_Y20_Z15"] = [0.5 * (xr[0] + xr[1]), 20.0, 15.0]
    d["points"] = {k: {"local_XYZ_mm": [r(v, 2) for v in p],
                       "displacement_local_XYZ_mm": [r(v) for v in displacement_local(site, tx, p)[0]]}
                   for k, p in pts.items()}
    p = tx.GetParameters()
    d["euler_rot_deg_LAS"] = [r(np.degrees(v), 4) for v in p[:3]]
    d["translation_at_centre_local_mm"] = [r(v) for v in
                                           (np.array(p[3:])[::-1] @ site.frame.R.T)]
    return d


def run(case):
    t0 = time.time()
    site = Site(case)
    frozen = json.loads(FROZEN.read_text())["cases"][case]["A"]
    vol = site.res["volume"]; ctrl = site.res["controls"]
    assert frozen["net_mm3"] == vol["PRIMARY_net_bone_gain_mm3"]
    area = float(vol["projected_area_mm2"])
    s_ctrl = float(ctrl["registration_accuracy"]["rigid_shift_mm"])
    dil_ctrl = float(ctrl["registration_accuracy"]["surface_dilation_mm"])
    roi = site.res["roi"]
    roi_centre = (0.5 * sum(roi["X_mm"]), 0.5 * sum(roi["Y_mm"]))

    sl = site.crop()
    print(f"[{case}] voxel {site.vx:.3f}, crop {[s.stop - s.start for s in sl]} "
          f"= {[(s.stop - s.start) * site.vx for s in sl]} mm", flush=True)
    pre = read_crop(site.pre_path, sl)
    pre_s = ndi.gaussian_filter(pre, geom.SMOOTH_SIGMA)
    print(f"  pre read, RSS {rss_gb():.2f} GB", flush=True)
    post = read_crop(site.post_path, sl)
    post_s = ndi.gaussian_filter(post, geom.SMOOTH_SIGMA)
    print(f"  post read, RSS {rss_gb():.2f} GB", flush=True)

    axes = site.tooth_axes(pre_s)
    out = {"case": case, "_script": f"analysis/site_registration_{TAG}.py", "voxel_mm": site.vx,
           "crop_lo_vox": site.lo.tolist(), "crop_shape_vox": list(pre.shape),
           "crop_mm": [r(n * site.vx, 2) for n in pre.shape],
           "bounding_teeth_fdi": site.res["config"]["bounding_teeth_fdi"],
           "pins_excluded_from_fixation_pins_json": site.pins,
           "tooth_axes_local": {t: {"cej_centre_XYZ_mm": [r(v, 2) for v in a["cej_centre"]],
                                    "apex_XYZ_mm": [r(v, 2) for v in a["apex"]],
                                    "source": a["source"], "cej_run_width_mm": a["cej_run_width_mm"]}
                                for t, a in axes.items()},
           "roi": {"X_mm": roi["X_mm"], "Y_mm": roi["Y_mm"], "centre_XY_mm": list(roi_centre),
                   "measured_area_mm2": area, "source": "results.json volume.projected_area_mm2"},
           "frozen": {"net_mm3": frozen["net_mm3"],
                      "gain_before_background_mm3": vol["gain_before_background_and_implant_mm3"],
                      "background_mm": ctrl["background_offset_mm"],
                      "control_rigid_shift_mm": s_ctrl, "control_dilation_mm": dil_ctrl,
                      "control_windows": site.res["control_windows"],
                      "per_window": {nm: {k: w.get(k) for k in ("rigid_shift_mm", "dilation_mm", "valid")}
                                     for nm, w in ctrl["windows"].items()}}}

    B = {}
    for T in (T_PRIMARY,) + T_SENS:
        rows = method_B(site, axes, pre_s, post_s, T)
        B[f"T{int(T)}"] = summarise_B(rows, roi_centre)
        c = B[f"T{int(T)}"].get("crown") or {}
        print(f"  B T{int(T)}: {len(rows)} rays, crown plane@ROI {c.get('plane_at_roi_centre_mm')} "
              f"X-interp {c.get('X_interp_of_tooth_medians_mm')}", flush=True)
    out["B_tooth_surface_decomposition"] = B

    masks, minfo = build_masks(site, axes, pre, post, pre_s, post_s)
    out["A_mask"] = minfo
    print(f"  masks {minfo['joint_voxels']} voxels, RSS {rss_gb():.2f} GB", flush=True)
    fixed = to_img(pre, site)
    moving = to_img(post, site)
    centre = 0.5 * (axes["mesial"]["cej_centre"] + axes["distal"]["cej_centre"])
    centre[1] = 20.0
    centre[2] = 15.0
    A = {"rotation_centre_local_mm": [r(v, 2) for v in centre], "runs": {}}
    starts = [(0.0, 0.0, 0.0)] + [tuple(float(s) * np.eye(3)[k]) for k in range(3) for s in (-0.5, 0.5)]
    for st in starts:
        tx, info = register(site, fixed, moving, masks["joint"], centre, st)
        key = ("joint_identity" if st == (0.0, 0.0, 0.0)
               else "joint_start_local_XYZ_%+.1f_%+.1f_%+.1f" % tuple(float(v) + 0.0 for v in st))
        A["runs"][key] = dict(info, **describe_tx(site, tx, axes))
        print(f"  A {key}: ROI buccal {A['runs'][key]['roi_Z15']['buccal_mean_mm']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    for t in ("mesial", "distal"):
        xa, za = site.axis_at(axes[t], 20.0)
        tx, info = register(site, fixed, moving, masks[t], np.array([xa, 20.0, 15.0]))
        A["runs"][f"{t}_only_identity"] = dict(info, **describe_tx(site, tx, axes))
    pm = A["runs"]["mesial_only_identity"]["points"]["mesial_axis_Y20_Z15"]
    pd_ = A["runs"]["distal_only_identity"]["points"]["distal_axis_Y20_Z15"]
    xm, xd = pm["local_XYZ_mm"][0], pd_["local_XYZ_mm"][0]
    um, ud = pm["displacement_local_XYZ_mm"][2], pd_["displacement_local_XYZ_mm"][2]
    A["per_tooth_X_interp_at_roi_centre_buccal_mm"] = r(um + (roi_centre[0] - xm) / (xd - xm) * (ud - um))
    bj = [v["roi_Z15"]["buccal_mean_mm"] for k, v in A["runs"].items() if k.startswith("joint")]
    A["joint_restart_buccal_range_mm"] = [min(bj), max(bj)]
    A["primary_roi_buccal_mean_mm"] = A["runs"]["joint_identity"]["roi_Z15"]["buccal_mean_mm"]
    out["A_rigid_reregistration_on_teeth"] = A
    print(f"  A per-tooth X-interp {A['per_tooth_X_interp_at_roi_centre_buccal_mm']}", flush=True)

    u_loc = np.array([0.0, 0.0, VALIDATION_SHIFT_MM])
    u_idx = (u_loc @ site.frame.R) / site.vx
    grid = np.indices(post.shape, dtype=np.float32)
    for k in range(3):
        grid[k] -= np.float32(u_idx[k])
    post_sh = ndi.map_coordinates(post, grid, order=1, mode="nearest").astype(np.float32)
    del grid
    post_sh_s = ndi.gaussian_filter(post_sh, geom.SMOOTH_SIGMA)
    V = {"applied_buccal_shift_mm": VALIDATION_SHIFT_MM}
    tx, info = register(site, fixed, to_img(post_sh, site), masks["joint"], centre)
    dA = describe_tx(site, tx, axes)["roi_Z15"]["buccal_mean_mm"]
    V["A_recovered_roi_buccal_mm"] = dA
    V["A_increment_over_unshifted_mm"] = r(dA - A["primary_roi_buccal_mean_mm"])
    rows = method_B(site, axes, pre_s, post_sh_s, T_PRIMARY)
    sB = summarise_B(rows, roi_centre, n_boot=200)
    c0 = B[f"T{int(T_PRIMARY)}"]["crown"]; c1 = sB["crown"]
    V["B_crown_plane_increment_mm"] = r(c1["plane_at_roi_centre_mm"] - c0["plane_at_roi_centre_mm"])
    V["B_crown_Xinterp_increment_mm"] = r(c1["X_interp_of_tooth_medians_mm"] - c0["X_interp_of_tooth_medians_mm"])
    V["B_crown_rays_shifted"] = c1["n_rays"]
    out["positive_control"] = V
    print(f"  validation: A +{V['A_increment_over_unshifted_mm']}  B plane "
          f"+{V['B_crown_plane_increment_mm']}  B X-interp +{V['B_crown_Xinterp_increment_mm']}", flush=True)
    del post_sh, post_sh_s

    cB = B[f"T{int(T_PRIMARY)}"]["crown"]
    est = {"A_joint_identity": A["primary_roi_buccal_mean_mm"],
           "A_per_tooth_X_interp": A["per_tooth_X_interp_at_roi_centre_buccal_mm"],
           "B_crown_plane_T1500": cB["plane_at_roi_centre_mm"],
           "B_crown_X_interp_T1500": cB["X_interp_of_tooth_medians_mm"]}
    ve = {}
    for k, d in est.items():
        ve[k] = {"site_buccal_offset_d_mm": d,
                 "raw_effect_d_x_area_mm3": r(d * area, 1),
                 "over_read_of_frozen_(d_minus_control_shift)_x_area_mm3": r((d - s_ctrl) * area, 1),
                 "net_with_site_shift_substituted_mm3": r(frozen["net_mm3"] - (d - s_ctrl) * area, 1)}
    ve["_definition"] = ("")
    out["volume_effect"] = ve
    out["peak_rss_GiB"] = r(rss_gb(), 2)
    out["runtime_s"] = r(time.time() - t0, 1)
    p = Path(f"analysis/{case}/site_registration_{TAG}.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}  RSS {rss_gb():.2f} GB  {time.time() - t0:.0f}s")


def collect():
    fin = {"_script": f"analysis/site_registration_{TAG}.py",
           "_question": "",
           "cases": {}, "summary_table": []}
    for case in ("S4", "S2"):
        o = json.loads(Path(f"analysis/{case}/site_registration_{TAG}.json").read_text())
        fin["cases"][case] = o
        for k, v in o["volume_effect"].items():
            if k.startswith("_"):
                continue
            fin["summary_table"].append({"case": case, "estimator": k, **v,
                                         "control_rigid_shift_mm": o["frozen"]["control_rigid_shift_mm"],
                                         "frozen_net_mm3": o["frozen"]["net_mm3"],
                                         "area_mm2": o["roi"]["measured_area_mm2"]})
    OUT_FINAL.write_text(json.dumps(fin, indent=1))
    print(f"wrote {OUT_FINAL}")


if __name__ == "__main__":
    if sys.argv[1] == "--collect":
        collect()
    else:
        run(sys.argv[1])

