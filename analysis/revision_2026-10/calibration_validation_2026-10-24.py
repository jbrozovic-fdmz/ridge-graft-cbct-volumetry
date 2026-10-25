#!/usr/bin/env python3
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path.cwd()
SFXD = ROOT / "analysis/scalefix_2026-10-06"
sys.path.insert(0, str(SFXD))
import sfx_measure as SM
from sfx_common import geom, V, READERS, post_path, sha256_file, write_new, rss_gib
from pipeline import calibrate

OUT = Path(__file__).resolve().parent
TAG = "2026-10-24"
PRIMARY = {"S1": "similarity_height_metalctl", "S2": "affine_height_metalctl", "S3": "similarity_height",
           "S4": "similarity_height_metalctl"}
STORED_PRIMARY = {"S1": (99.7, 99.5), "S3": (137.0, 138.3), "S2": (95.4, 83.6), "S4": (267.1, 271.7)}
PAPER = {"S1": "S1", "S3": "X1", "S2": "S2", "S4": "X2"}
DEPTH_JSON = ROOT / "FINAL_ARCH_20260907/containment_NULLCORRECTED.json"
L0 = 450.0
NPTS = 400000
T = np.round(np.arange(-3.0, 8.0001, 0.025), 4)


def load_full(p):
    img = sitk.ReadImage(str(p), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(img), float(img.GetSpacing()[0])


def primary_record(case):
    tag = PRIMARY[case]
    mj = SFXD / case / f"measure_{tag}.json"
    return mj, json.loads(mj.read_text())


def arch_model(R, pre, vx, fov=None):
    lm = R["config"]["landmarks_preop_SAL_mm"]
    cej_S = 0.5 * (lm["mesial_tooth_cej_buccal"][0] + lm["distal_tooth_cej_buccal"][0])
    sg = 1.0 if R["frame"]["jaw"] == "maxilla" else -1.0
    s_lo, s_hi = sorted([cej_S - sg * 6.0, cej_S - sg * 1.0])
    k0, k1 = max(int(s_lo / vx), 0), min(int(s_hi / vx), pre.shape[0])
    t2 = (pre[k0:k1] > 2500)
    if fov is not None:
        t2 &= fov[k0:k1]
    t2 = ndi.binary_opening(t2.any(0), iterations=2)
    ij = np.argwhere(t2) * vx
    cA, cL = ij.mean(0)
    ang = np.degrees(np.arctan2(ij[:, 1] - cL, ij[:, 0] - cA)); rad = np.hypot(ij[:, 0] - cA, ij[:, 1] - cL)
    bins = np.arange(-180, 181, 3.0); Rb = np.full(len(bins) - 1, np.nan)
    for b in range(len(bins) - 1):
        s = (ang >= bins[b]) & (ang < bins[b + 1])
        if s.sum() > 30:
            Rb[b] = np.median(rad[s])
    bc = 0.5 * (bins[1:] + bins[:-1])
    return {"cA": float(cA), "cL": float(cL), "bins": bc, "R": Rb, "crown_slab_S_mm": [s_lo, s_hi]}


def maxilla_keep(fr, P_SAL_mm, am, y_min=7.0, lateral_max=12.0):
    loc = fr.to_local(P_SAL_mm)
    ang = np.degrees(np.arctan2(P_SAL_mm[:, 2] - am["cL"], P_SAL_mm[:, 1] - am["cA"]))
    rad = np.hypot(P_SAL_mm[:, 1] - am["cA"], P_SAL_mm[:, 2] - am["cL"])
    ok = np.isfinite(am["R"])
    Ra = np.interp(ang, am["bins"][ok], am["R"][ok], left=np.nan, right=np.nan)
    return (loc[:, 1] > y_min) & np.isfinite(Ra) & (rad < Ra + lateral_max)


def grey_pairs(case, pre, post, vx, R, fr, ext):
    fov = ndi.binary_erosion((post > -999) & (pre > -1150), iterations=int(round(3 / vx)))
    gs = ndi.gaussian_gradient_magnitude(pre, 1.0) / vx
    flat = gs < np.percentile(gs[fov], 40)
    del gs
    idx = np.argwhere(fov & flat); del fov, flat
    rng = np.random.default_rng(3); idx = idx[rng.choice(len(idx), min(len(idx), 3_000_000), replace=False)]
    P = idx * vx; loc = fr.to_local(P)
    am = arch_model(R, pre, vx)
    keep = ~((loc[:, 0] > ext[0] - 8) & (loc[:, 0] < ext[1] + 8) & (loc[:, 2] > -5) & (loc[:, 2] < 60))
    keep &= maxilla_keep(fr, P, am, lateral_max=25.0)
    a = pre[tuple(idx[keep].T)]; b = post[tuple(idx[keep].T)]
    bins = np.array([-1100, -900, -700, -300, 0, 150, 300, 450, 600, 800, 1000, 1250, 1500, 2000, 2500, 3000, 4000, 6000])
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (a >= lo) & (a < hi)
        if m.sum() < 200:
            continue
        rows.append({"pre_bin": [int(lo), int(hi)], "n": int(m.sum()), "pre_median": round(float(np.median(a[m])), 1),
                     "post_median": round(float(np.median(b[m])), 1),
                     "post_IQR": [round(float(v), 1) for v in np.percentile(b[m], [25, 75])]})
    m = (a > 300) & (a < 2500)
    sl, ic = np.polyfit(a[m], b[m], 1)
    boot = []
    rb = np.random.default_rng(11)
    am_, bm_ = a[m], b[m]
    blk = np.floor(idx[keep][m] * vx / 5.0).astype(int)
    _, bid = np.unique(blk, axis=0, return_inverse=True); nb = bid.max() + 1
    for _ in range(200):
        pick = rb.integers(0, nb, nb); w = np.bincount(pick, minlength=nb)[bid].astype(float)
        s2, i2 = np.polyfit(am_, bm_, 1, w=np.sqrt(w))
        boot.append(s2 * 450 + i2)
    return {"n_voxels": int(keep.sum()), "rows": rows,
            "bone_range_fit_post_eq_a_pre_plus_b": [round(float(sl), 4), round(float(ic), 1)],
            "post_value_where_pre_is_450": round(float(sl * 450 + ic), 1),
            "post_value_where_pre_is_450_ci95_block_bootstrap": [round(float(v), 1) for v in np.percentile(boot, [2.5, 97.5])],
            "grey_offset_at_450": round(float(sl * 450 + ic - 450.0), 1)}


def crossings(f, ref, level):
    a, b = f[:, :-1], f[:, 1:]
    c = (a >= level) & (b < level)
    frac = np.where(c, (a - level) / np.where(a - b == 0, 1, a - b), np.nan)
    tc = np.where(c, T[:-1] + frac * 0.025, np.nan)
    near = np.full(len(f), np.nan); outer = np.nanmax(np.where(c, tc, -np.inf), axis=1)
    outer[~np.isfinite(outer)] = np.nan
    dist = np.abs(tc - ref[:, None]); dist[~c] = np.inf
    j = np.argmin(dist, axis=1); hit = np.isfinite(dist[np.arange(len(f)), j])
    near[hit] = tc[np.arange(len(f)), j][hit]
    return near, outer


def surface_profiles(pre, post, vx, R, fr):
    lm = R["config"]["landmarks_preop_SAL_mm"]
    fov = ndi.binary_erosion((post > -999.0) & (pre > -1150.0), iterations=int(round(3.0 / vx)))

    def _sat(v):
        mx = float(v.max()); n = int((v >= mx - 1.0).sum())
        return (v >= mx - 1.0) if n > 1000 else np.zeros(v.shape, bool)
    metal = ndi.binary_dilation((pre > 7000) | (post > 7000) | _sat(pre) | _sat(post), iterations=int(round(2.0 / vx)))
    am = arch_model(R, pre, vx, fov=fov)
    fpre = ndi.gaussian_filter(pre, geom.SMOOTH_SIGMA); fpost = ndi.gaussian_filter(post, geom.SMOOTH_SIGMA)
    inside = fpre >= L0
    surf = inside & ~ndi.binary_erosion(inside) & fov & ~metal
    del inside, metal, fov
    idx = np.argwhere(surf); del surf
    rng = np.random.default_rng(20261006)
    if len(idx) > NPTS:
        idx = idx[np.sort(rng.choice(len(idx), NPTS, replace=False))]
    P = idx.astype(float)
    g = np.zeros_like(P)
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        g[:, k] = (ndi.map_coordinates(fpre, (P + e).T, order=1) - ndi.map_coordinates(fpre, (P - e).T, order=1)) / (2 * vx)
    gn = np.linalg.norm(g, axis=1); good = gn > 50.0
    P, g, gn = P[good], g[good], gn[good]
    N = -g / gn[:, None]
    tpre = np.empty(len(P)); inmax = np.empty(len(P))
    FB = np.empty((len(P), len(T)), np.float32)
    for s in range(0, len(P), 15000):
        p, n = P[s:s + 15000], N[s:s + 15000]
        Q = p[:, None, :] + (T[None, :, None] / vx) * n[:, None, :]
        fa = ndi.map_coordinates(fpre, Q.reshape(-1, 3).T, order=1).reshape(len(p), len(T))
        fb = ndi.map_coordinates(fpost, Q.reshape(-1, 3).T, order=1).reshape(len(p), len(T))
        tp, _ = crossings(fa, np.zeros(len(p)), L0)
        tpre[s:s + 15000] = tp
        FB[s:s + 15000] = fb
        ins = (T < 0) & (T > -1.5)
        raw = ndi.map_coordinates(pre, Q[:, ins].reshape(-1, 3).T, order=1).reshape(len(p), ins.sum())
        inmax[s:s + 15000] = raw.max(1)
    del fpre, fpost
    Pmm = P * vx
    radp = np.hypot(Pmm[:, 1] - am["cA"], Pmm[:, 2] - am["cL"])
    radial = np.stack([np.zeros(len(P)), (Pmm[:, 1] - am["cA"]) / radp, (Pmm[:, 2] - am["cL"]) / radp], 1)
    facing = np.sum(N * radial, 1)
    return {"P": Pmm, "N": N, "tpre": tpre, "FB": FB, "tooth": inmax > 3000, "facing": facing, "am": am}


NAMES = ["sigma", "t_S", "t_A", "t_L", "w_S", "w_A", "w_L", "delta_tooth", "delta_bone"]


def near_crossing(FB, ref, level, chunk=20000):
    out = np.empty(len(FB))
    for s in range(0, len(FB), chunk):
        out[s:s + chunk], _ = crossings(FB[s:s + chunk], ref[s:s + chunk], level)
    return out


def fit_model(S, fr, ext, level, boot=0):
    tn = near_crossing(S["FB"], np.nan_to_num(S["tpre"]), level)
    d = tn - S["tpre"]
    loc = fr.to_local(S["P"])
    graftseg = (loc[:, 0] > ext[0] - 8) & (loc[:, 0] < ext[1] + 8) & (loc[:, 2] > -5) & (loc[:, 2] < 60)
    MAXI = maxilla_keep(fr, S["P"], S["am"])
    far = ~graftseg & np.isfinite(d) & (np.abs(d) < 2.0) & MAXI
    P, N, tooth = S["P"][far], S["N"][far], S["tooth"][far]
    c = P.mean(0)
    r = P - c
    cr = np.cross(r, N)
    A = np.column_stack([np.sum(N * r, 1), N[:, 0], N[:, 1], N[:, 2], cr[:, 0], cr[:, 1], cr[:, 2],
                         tooth.astype(float), (~tooth).astype(float)])
    y = d[far]; w = np.ones(len(y))
    for _ in range(20):
        sw = np.sqrt(w); beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        res = y - A @ beta; s = 1.4826 * np.median(np.abs(res)); u = res / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    fac = S["facing"][far]
    out = {"level_healed": level, "n_far": int(far.sum()), "beta": {k: float(beta[i]) for i, k in enumerate(NAMES)},
           "robust_sigma_resid_mm": round(float(s), 4),
           "far_bone_outer_median_mm": round(float(np.median(y[~tooth & (fac > 0.5)])), 4),
           "far_bone_inner_median_mm": round(float(np.median(y[~tooth & (fac < -0.5)])), 4)}
    if boot:
        rng = np.random.default_rng(1); blk = np.floor(P / 5.0).astype(int)
        _, bid = np.unique(blk, axis=0, return_inverse=True); nb = bid.max() + 1; bs = []
        for _ in range(boot):
            pick = rng.integers(0, nb, nb); wb = np.bincount(pick, minlength=nb)[bid] * w
            swb = np.sqrt(wb); bb, *_ = np.linalg.lstsq(A * swb[:, None], y * swb, rcond=None); bs.append(bb)
        bs = np.array(bs)
        out["ci95"] = {k: [round(float(v), 5) for v in np.percentile(bs[:, i], [2.5, 97.5])] for i, k in enumerate(NAMES)}
    return out


def stage_grey(case):
    outp = OUT / f"CALVAL_GREY_{case}_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    t0 = time.time()
    mj, M = primary_record(case)
    post = M["post"]
    R = json.loads((ROOT / f"analysis/{case}/results.json").read_text())
    fr = geom.LocalFrame.from_dict(R["frame"])
    ge = M["readers"]["A"]["graft_extent"]
    ext = (float(ge["x_from_mm"]), float(ge["x_to_mm"]))
    pre, vx = load_full(ROOT / f"analysis/{case}/pre_cal.nii.gz")
    pst, vx2 = load_full(post_path(case, post))
    assert abs(vx - vx2) < 1e-9
    g = grey_pairs(case, pre, pst, vx, R, fr, ext)
    print(case, "grey", g["bone_range_fit_post_eq_a_pre_plus_b"], g["post_value_where_pre_is_450"], flush=True)
    S = surface_profiles(pre, pst, vx, R, fr)
    del pre, pst
    m450 = fit_model(S, fr, ext, L0, boot=200)
    db = lambda lv: fit_model(S, fr, ext, lv)["beta"]["delta_bone"]
    grid = {f"{lv:.0f}": round(db(lv), 4) for lv in np.arange(350.0, 651.0, 50.0)}
    lo, hi = 250.0, 800.0
    flo, fhi = db(lo), db(hi)
    root = None
    if flo > 0 > fhi:
        for _ in range(30):
            mid = 0.5 * (lo + hi); fm = db(mid)
            if fm > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.25:
                break
        root = round(0.5 * (lo + hi), 1)
    out = {"_what": "",
           "case": case, "paper_code": PAPER[case], "primary_tag": PRIMARY[case], "post": post,
           "graft_extent_excluded_local_X_mm": [ext[0] - 8, ext[1] + 8], "voxel_paired_grey": g,
           "surface_displacement_450": {"model": {k: round(v, 5) for k, v in m450["beta"].items()},
                                        "ci95_block_bootstrap": m450["ci95"], "n_far": m450["n_far"],
                                        "robust_sigma_resid_mm": m450["robust_sigma_resid_mm"],
                                        "far_bone_outer_facing_median_mm": m450["far_bone_outer_median_mm"],
                                        "far_bone_inner_facing_median_mm": m450["far_bone_inner_median_mm"],
                                        "n_surface_points": int(len(S["P"]))},
           "delta_bone_vs_healed_level_mm": grid,
           "healed_level_with_zero_bone_dilation": root,
           "inputs": {str(mj.relative_to(ROOT)): sha256_file(mj),
                      str(post_path(case, post).relative_to(ROOT)): sha256_file(post_path(case, post)),
                      f"analysis/{case}/pre_cal.nii.gz": sha256_file(ROOT / f"analysis/{case}/pre_cal.nii.gz"),
                      f"analysis/{case}/results.json": sha256_file(ROOT / f"analysis/{case}/results.json")},
           "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256_file(Path(__file__).resolve()),
           "peak_rss_GiB": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    print(case, "dilation", out["surface_displacement_450"]["model"]["delta_bone"], m450["ci95"]["delta_bone"],
          "zero-dilation level", root, flush=True)


def controls_at(c, R, P, Q, MR):
    z0, z1 = c.bands[R]; KEEP = c.KEEP[R]
    meds, wins, flags = [], {}, []
    for nm, w in MR["controls"]["windows"].items():
        if not w.get("valid"):
            continue
        b = V.measure_window(P, Q, w["X_mm"], w["Y_mm"], SM.GRID, z0, z1, keep_box=KEEP)
        if b.get("cells", 0) == 0:
            flags.append(f"{nm}: no cells"); continue
        bo = b["median_thickness_mm"]
        if b["cells"] < 200:
            flags.append(f"{nm}: only {b['cells']} cells")
        if abs(bo) > SM.TOL_GATE:
            flags.append(f"{nm}: buccal median {bo:+.3f} mm exceeds {SM.TOL_GATE}")
        meds.append(bo); wins[nm] = {"buccal_median_mm": bo, "cells": b["cells"]}
    return (float(np.mean(meds)) if meds else None), wins, flags


def window_net(c, R, P, Q, MR):
    roi = c.res[R]["roi"]; z0, z1 = c.bands[R]; KEEP = c.KEEP[R]
    m0 = V.measure_window(P, Q, roi["X_mm"], roi["Y_mm"], SM.GRID, z0, z1, background=0.0, keep_box=KEEP)
    Hpre, Hpost, xe, ye = m0["_Hpre"], m0["_Hpost"], m0["_x_edges"], m0["_y_edges"]
    bg, wins, flags = controls_at(c, R, P, Q, MR)
    imp_t, _ = SM.ti_term(c, R, c.implant_idx, Hpre, Hpost, xe, ye)
    prom = {L: bool(MR["prominence"][str(L)]["cap_applied"]) for L in c.elig}
    Hc, _, _, _ = SM.d7e_surface(c, R, xe, ye, Hpre, Hpost, prom, SM.D_PRIMARY)
    m = V.measure_window(None, None, None, None, SM.GRID, background=bg, fields=(Hpre, Hc, xe, ye))
    pin_t, _ = SM.ti_term(c, R, c.pin_idx, Hpre, Hc, xe, ye)
    main = V.measure_window(None, None, None, None, SM.GRID, background=bg, fields=(Hpre, Hpost, xe, ye))
    return {"net_mm3": round(m["gain_mm3"] - imp_t - pin_t, 1), "gross_mm3": main["gain_mm3"], "implant_mm3": imp_t,
            "pin_metal_mm3": pin_t, "background_mm": round(bg, 4), "background_unrounded": bg,
            "coverage": m["coverage"], "area_mm2": m["area_mm2"], "controls": wins, "qc_flags": flags}


def clouds(c, field, level):
    Vx, Fx = geom.iso_surface(field, c.vx, level=level); Vx = Vx + c.origin_mm
    return {R: geom.surface_points_local(Vx, Fx, c.frames[R], c.KEEP[R]) for R in READERS}


def alternatives(cal):
    peaks = [float(p) for p in cal["histogram_peaks"]]
    alts, excl = [], []
    if len(peaks) <= 2:
        return alts, excl
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            air, soft = round(peaks[i], 1), round(peaks[j], 1)
            if air == cal["air_grey"] and soft == cal["soft_tissue_grey"]:
                continue
            span = soft - air
            rec = {"air": air, "soft": soft, "span": round(span, 1)}
            if calibrate.MIN_SEPARATION <= span <= calibrate.MAX_SEPARATION:
                alts.append(rec)
            else:
                rec["excluded_because"] = (f"air-to-soft span {span:.1f} outside the calibration's own admissible range "
                                           f"{calibrate.MIN_SEPARATION:.0f}-{calibrate.MAX_SEPARATION:.0f}")
                rec["native_equivalent_of_iso450"] = round((450.0 - calibrate.AIR_HU) * span / (calibrate.SOFT_HU - calibrate.AIR_HU) + air, 1)
                excl.append(rec)
    return alts, excl


def stage_volume(case):
    outp = OUT / f"CALVAL_VOLUME_{case}_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    t0 = time.time()
    G = json.loads((OUT / f"CALVAL_GREY_{case}_{TAG}.json").read_text())
    mj, M = primary_record(case)
    post = M["post"]
    c = SM.setup(case, post, str(DEPTH_JSON))
    c.surfaces = None; c.P = None; c.Q = None
    pf = geom.bone_field(c.pre_c); qf = geom.bone_field(c.post_c)
    P450 = clouds(c, pf, L0); Q450 = clouds(c, qf, L0)
    base = {R: window_net(c, R, P450[R], Q450[R], M["readers"][R]) for R in READERS}
    gate = {R: {"net": [base[R]["net_mm3"], M["readers"][R]["primary"]["net_mm3"], STORED_PRIMARY[case][i]],
                "background_unrounded": [base[R]["background_unrounded"], M["readers"][R]["controls"]["background_unrounded"]]}
            for i, R in enumerate(READERS)}
    passed = all(v[0] == x for R in READERS for v in gate[R].values() for x in v[1:])
    if not passed:
        raise SystemExit(f"{case}: regression gate failed {gate}")
    res = {"gate_450": {"passed": passed, "values": gate}}
    levels = {"grey_matched": G["voxel_paired_grey"]["post_value_where_pre_is_450"],
              "geometry_matched": G["healed_level_with_zero_bone_dilation"]}
    res["matched_threshold"] = {}
    for nm, lv in levels.items():
        if lv is None:
            res["matched_threshold"][nm] = {"level_healed": None}
            continue
        Qm = clouds(c, qf, float(lv))
        r_ = {R: window_net(c, R, P450[R], Qm[R], M["readers"][R]) for R in READERS}
        res["matched_threshold"][nm] = {"level_healed": lv, "readers": r_,
                                        "delta_net_mm3": {R: round(r_[R]["net_mm3"] - base[R]["net_mm3"], 1) for R in READERS}}
        del Qm
    dil = G["surface_displacement_450"]["model"]["delta_bone"]
    ci = G["surface_displacement_450"]["ci95_block_bootstrap"]["delta_bone"]
    res["bound_dilation_times_area"] = {R: {"area_mm2": base[R]["area_mm2"], "dilation_mm": dil,
                                            "delta_volume_mm3": round(dil * base[R]["area_mm2"], 1),
                                            "ci95_mm3": [round(ci[0] * base[R]["area_mm2"], 1), round(ci[1] * base[R]["area_mm2"], 1)]}
                                        for R in READERS}
    R0 = json.loads((ROOT / f"analysis/{case}/results.json").read_text())
    res["alternative_peaks"] = {}
    for which in ("pre", "post"):
        cal = R0["calibration"][which]
        alts, excl = alternatives(cal)
        if not alts and not excl:
            continue
        src = ROOT / cal["source"]
        img = sitk.ReadImage(str(src))
        a = sitk.GetArrayFromImage(img).astype(np.float32); del img
        ref0 = {"air": cal["air_grey"], "soft": cal["soft_tissue_grey"], "padding_value": cal["padding_value"]}
        rec = {"histogram_peaks": cal["histogram_peaks"], "used": {"air": cal["air_grey"], "soft": cal["soft_tissue_grey"]},
               "excluded_assignments": excl, "runs": []}
        if which == "pre":
            b0 = calibrate.calibrate_array(a, ref0)[c.sl].astype(np.float32)
            rec["reconstruction_of_stored_calibrated_crop_bit_identical"] = bool(np.array_equal(b0, c.pre_c))
            assert rec["reconstruction_of_stored_calibrated_crop_bit_identical"]
            del b0
            for alt in alts:
                ref = {"air": alt["air"], "soft": alt["soft"], "padding_value": cal["padding_value"]}
                b = calibrate.calibrate_array(a, ref)[c.sl].astype(np.float32)
                Pa = clouds(c, geom.bone_field(b), L0); del b
                r_ = {R: window_net(c, R, Pa[R], Q450[R], M["readers"][R]) for R in READERS}
                scale = (calibrate.SOFT_HU - calibrate.AIR_HU) / (alt["soft"] - alt["air"])
                rec["runs"].append({"assignment": alt, "native_equivalent_of_iso450": round((450.0 - calibrate.AIR_HU) / scale + alt["air"], 1),
                                    "readers": r_, "delta_net_mm3": {R: round(r_[R]["net_mm3"] - base[R]["net_mm3"], 1) for R in READERS},
                                    "registration": ""})
                del Pa
        else:
            rec["runs_note"] = ""
        res["alternative_peaks"][which] = rec
        del a
    out = {"_what": "", "case": case, "paper_code": PAPER[case], "primary_tag": PRIMARY[case],
           "base_450": base, **res,
           "inputs": {str(mj.relative_to(ROOT)): sha256_file(mj), f"CALVAL_GREY_{case}_{TAG}.json": sha256_file(OUT / f"CALVAL_GREY_{case}_{TAG}.json")},
           "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256_file(Path(__file__).resolve()),
           "peak_rss_GiB": round(rss_gib(), 2), "runtime_s": round(time.time() - t0, 1),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    print(case, json.dumps({k: out[k] for k in ("gate_450", "bound_dilation_times_area")}), flush=True)
    print(case, {nm: v.get("delta_net_mm3") for nm, v in out["matched_threshold"].items()}, flush=True)


def collect():
    outp = OUT / f"CALIBRATION_VALIDATION_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    rows, per = [], {}
    for c in ("S1", "S3", "S2", "S4"):
        G = json.loads((OUT / f"CALVAL_GREY_{c}_{TAG}.json").read_text())
        Vv = json.loads((OUT / f"CALVAL_VOLUME_{c}_{TAG}.json").read_text())
        alt = Vv.get("alternative_peaks", {}).get("pre", {}).get("runs", [])
        row = {"case": PAPER[c],
               "grey_offset_at_450": G["voxel_paired_grey"]["grey_offset_at_450"],
               "healed_grey_where_pre_is_450": G["voxel_paired_grey"]["post_value_where_pre_is_450"],
               "healed_grey_where_pre_is_450_ci95": G["voxel_paired_grey"]["post_value_where_pre_is_450_ci95_block_bootstrap"],
               "dilation_mm": G["surface_displacement_450"]["model"]["delta_bone"],
               "dilation_ci95_mm": G["surface_displacement_450"]["ci95_block_bootstrap"]["delta_bone"],
               "delta_volume_a_bound_mm3": {R: Vv["bound_dilation_times_area"][R]["delta_volume_mm3"] for R in READERS},
               "delta_volume_b_grey_matched_mm3": Vv["matched_threshold"]["grey_matched"].get("delta_net_mm3"),
               "healed_level_grey_matched": Vv["matched_threshold"]["grey_matched"]["level_healed"],
               "delta_volume_b_geometry_matched_mm3": Vv["matched_threshold"]["geometry_matched"].get("delta_net_mm3"),
               "healed_level_geometry_matched": Vv["matched_threshold"]["geometry_matched"]["level_healed"],
               "delta_volume_alt_peaks_mm3": [{"assignment": r["assignment"], "delta_net_mm3": r["delta_net_mm3"]} for r in alt] or None,
               "gate_450_passed": Vv["gate_450"]["passed"]}
        rows.append(row)
        per[PAPER[c]] = {"grey_file": f"CALVAL_GREY_{c}_{TAG}.json", "volume_file": f"CALVAL_VOLUME_{c}_{TAG}.json",
                         "grey_sha256": sha256_file(OUT / f"CALVAL_GREY_{c}_{TAG}.json"),
                         "volume_sha256": sha256_file(OUT / f"CALVAL_VOLUME_{c}_{TAG}.json"),
                         "peak_rss_GiB": {"grey": G["peak_rss_GiB"], "volume": Vv["peak_rss_GiB"]}}
    out = {"_what": "", "table": rows, "per_case_files": per,
           "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256_file(Path(__file__).resolve()),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    f = lambda d: "/".join(f"{d[R]:+.1f}" for R in READERS) if d else "n/a"
    lines = ["| case | healed grey at pre 450 (offset) | dilation mm [95 % CI] | Δ volume (a) A/B | "
             "Δ volume (b) grey-matched A/B | Δ volume (b) geometry-matched A/B | Δ volume alt. peaks A/B |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        alt = "; ".join(f"air {a['assignment']['air']}, soft {a['assignment']['soft']}: {f(a['delta_net_mm3'])}"
                        for a in r["delta_volume_alt_peaks_mm3"]) if r["delta_volume_alt_peaks_mm3"] else "n/a (2 peaks)"
        lines.append(f"| {r['case']} | {r['healed_grey_where_pre_is_450']} ({r['grey_offset_at_450']:+.1f}) | "
                     f"{r['dilation_mm']:+.3f} [{r['dilation_ci95_mm'][0]:+.3f}, {r['dilation_ci95_mm'][1]:+.3f}] | "
                     f"{f(r['delta_volume_a_bound_mm3'])} | {f(r['delta_volume_b_grey_matched_mm3'])} "
                     f"(level {r['healed_level_grey_matched']}) | {f(r['delta_volume_b_geometry_matched_mm3'])} "
                     f"(level {r['healed_level_geometry_matched']}) | {alt} |")
    with open(OUT / f"CALIBRATION_VALIDATION_{TAG}.md", "x") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == "--case" and a[1] in PRIMARY and a[2] == "--stage" and a[3] in ("grey", "volume"):
        (stage_grey if a[3] == "grey" else stage_volume)(a[1])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit("usage: --case S1|S2|S3|S4 --stage grey|volume | --collect")

