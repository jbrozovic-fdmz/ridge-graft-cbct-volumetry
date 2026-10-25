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
STEP = 0.20
DS = 0.05
PAD_VOX = 8
MAX_CROP_MM = 40.0
CHUNK_X_MM = 10.0
ISO = geom.ISO_LEVEL
METAL = geom.METAL_LEVEL
MIN_CONTRAST = 300.0
LEVELS = [float(v) for v in range(300, 701, 25)]
SEED = 20260913
NBOOT = 2000
OUT_FINAL = Path(f"FINAL_ARCH_20260907/cross_scanner_grey_{TAG}.json")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def rd(x, n=3):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


def edges(lo, hi, step=STEP):
    return np.arange(lo, hi + 1e-9, step)


def regions(res, case_cfg):
    roi = res["roi"]
    z0, z1 = [float(v) for v in res["search_band_Z_mm"]]
    lm = res["anchor"]["landmarks_local_XYZ_mm"]
    excl = set(case_cfg.get("exclude_control_windows") or [])
    R = {}
    R["roi_graft_NOT_stable"] = dict(X=roi["X_mm"], Y=roi["Y_mm"], z=(z0, z1), sign=+1,
                                     s_lim_z=z1, kind="roi", stable=False)
    ctrl_x = []
    for nm, xr in res["control_windows"].items():
        w = res["controls"]["windows"].get(nm, {})
        valid = bool(w.get("valid", True)) and nm not in excl
        R[f"control_{nm}_buccal_plate"] = dict(X=xr, Y=roi["Y_mm"], z=(z0, z1), sign=+1, s_lim_z=z1,
                                               kind="control", stable=valid,
                                               frozen_valid=valid,
                                               frozen_median_mm=w.get("buccal", {}).get(
                                                   "median_thickness_mm"))
        ctrl_x += list(xr)
    zl = float(lm["crest_lingual_midspan"][2])
    R["palatal_cortex"] = dict(X=[min(ctrl_x), max(ctrl_x)], Y=[16.0, 22.0],
                               z=(zl - 14.0, zl + 3.0), sign=-1, s_lim_z=zl - 14.0,
                               kind="palatal", stable=True)
    for t in ("mesial", "distal"):
        xc, yc, zc = [float(v) for v in lm[f"{t}_tooth_cej_buccal"]]
        R[f"tooth_{t}_crown_buccal"] = dict(X=[xc - 1.5, xc + 1.5], Y=[yc - 4.0, yc - 0.5],
                                            z=(zc - 5.0, zc + 5.0), sign=+1, s_lim_z=zc + 5.0,
                                            kind="tooth", stable=True)
    return R


def region_grid(reg):
    xe, ye = edges(*reg["X"]), edges(*reg["Y"])
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    z_lo, z_hi = reg["z"]
    if reg["sign"] > 0:
        zs = np.arange(z_lo - 2.5, z_hi + 2.5 + 1e-9, DS)
    else:
        zs = np.arange(z_lo - 2.5, z_hi + 2.5 + 1e-9, DS)
    return xe, ye, xc, yc, zs


def sample_scan(path, frame, R, vx):
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayViewFromImage(img)
    shape = np.array(a.shape)
    out = {}
    crops = []
    outside_count = {}
    for name, reg in R.items():
        xe, ye, xc, yc, zs = region_grid(reg)
        V = np.empty((len(yc), len(xc), len(zs)), np.float32)
        M = np.empty((len(yc), len(xc), len(zs)), bool)
        VR = np.empty((len(yc), len(xc), len(zs)), np.float32)
        nch = int(np.ceil((xc[-1] - xc[0] + STEP) / CHUNK_X_MM))
        for ch in np.array_split(np.arange(len(xc)), nch):
            X, Y, Z = np.meshgrid(xc[ch], yc, zs, indexing="ij")
            L = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            del X, Y, Z
            vox = frame.to_anatomical(L) / vx
            del L
            lo = np.maximum(np.floor(vox.min(0)).astype(int) - PAD_VOX, 0)
            hi = np.minimum(np.ceil(vox.max(0)).astype(int) + PAD_VOX + 1, shape)
            ext_mm = (hi - lo) * vx
            assert np.all(ext_mm <= MAX_CROP_MM + 1e-6), f"{name}: crop {ext_mm} mm > 40"
            inarr = np.all((vox >= 0) & (vox <= shape - 1), axis=1)
            outside_count[name] = outside_count.get(name, 0) + int((~inarr).sum())
            sl = tuple(slice(int(p), int(q)) for p, q in zip(lo, hi))
            crop = np.array(a[sl], dtype=np.float32)
            crops.append([round(float(e), 1) for e in ext_mm])
            field = geom.bone_field(crop)
            coords = (vox - lo).T
            v = ndi.map_coordinates(field, coords, order=1, mode="nearest")
            v[~inarr] = np.nan
            m = ndi.map_coordinates(crop, coords, order=1, mode="nearest") > METAL
            raw_s = crop.copy()
            ndi.gaussian_filter(raw_s, geom.SMOOTH_SIGMA, output=raw_s)
            vr = ndi.map_coordinates(raw_s, coords, order=1, mode="nearest")
            vr[~inarr] = np.nan
            del inarr, raw_s
            del field, crop, coords, vox
            nx_, ny_, nz_ = len(ch), len(yc), len(zs)
            V[:, ch, :] = v.reshape(nx_, ny_, nz_).transpose(1, 0, 2)
            M[:, ch, :] = m.reshape(nx_, ny_, nz_).transpose(1, 0, 2)
            VR[:, ch, :] = vr.reshape(nx_, ny_, nz_).transpose(1, 0, 2)
            del v, m, vr
        if reg["sign"] < 0:
            V = V[:, :, ::-1].copy()
            M = M[:, :, ::-1].copy()
            VR = VR[:, :, ::-1].copy()
            s = -zs[::-1]
        else:
            s = zs
        out[name] = dict(V=V, M=M, VR=VR, s=s, xc=xc, yc=yc, xe=xe, ye=ye,
                         outside_array_frac=round(outside_count.get(name, 0) / V.size, 4))
    del a, img
    return out, crops


def outer_crossing(V, s, level):
    N, ns = V.shape
    lev = np.broadcast_to(np.asarray(level, float), (N,))
    ge = V >= lev[:, None]
    anyge = ge.any(1)
    last = ns - 1 - np.argmax(ge[:, ::-1], axis=1)
    i = np.clip(last, 0, ns - 2)
    r = np.arange(N)
    v0 = V[r, i].astype(float)
    v1 = V[r, i + 1].astype(float)
    ok = anyge & (last < ns - 1) & np.isfinite(v1) & np.isfinite(v0) & np.isfinite(lev)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(ok, (v0 - lev) / (v0 - v1), np.nan)
    return np.where(ok, s[i] + t * (s[1] - s[0]), np.nan)


def analyse_scan(d, s_lim):
    V = d["V"].reshape(-1, d["V"].shape[-1]).astype(np.float64)
    M = d["M"].reshape(-1, d["M"].shape[-1])
    s = d["s"]
    Vi = np.where(s[None, :] <= s_lim + 1e-9, V, -np.inf)
    s_iso = outer_crossing(Vi, s, ISO)
    rel = s[None, :] - s_iso[:, None]
    with np.errstate(invalid="ignore"):
        omask = (rel >= 1.0) & (rel <= 2.0)
        imask = (rel >= -2.0) & (rel <= 0.0)
        wmask = (rel >= -2.0) & (rel <= 1.5)
        mmask = (rel >= -2.0) & (rel <= 2.0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            outer = np.nanmedian(np.where(omask, V, np.nan), axis=1)
    peak = np.max(np.where(imask, V, -np.inf), axis=1)
    peak = np.where(np.isfinite(peak), peak, np.nan)
    contrast = peak - outer
    Vw = np.where(wmask, V, -np.inf)
    s_hm = outer_crossing(Vw, s, 0.5 * (outer + peak))
    s10 = outer_crossing(Vw, s, outer + 0.1 * contrast)
    s90 = outer_crossing(Vw, s, outer + 0.9 * contrast)
    metal = np.any(M & mmask, axis=1)
    in_range = (s_iso - 2.0 >= s[0]) & (s_iso + 2.0 <= s[-1])
    valid = (np.isfinite(s_iso) & np.isfinite(s_hm) & in_range & np.isfinite(outer)
             & (contrast >= MIN_CONTRAST) & ~metal)
    sweep = {L: outer_crossing(Vi, s, L) for L in LEVELS}
    shp = d["V"].shape[:2]
    VRi = np.where(s[None, :] <= s_lim + 1e-9, d["VR"].reshape(-1, d["VR"].shape[-1]).astype(np.float64),
                   -np.inf)
    s_iso_raw = outer_crossing(VRi, s, ISO)
    del VRi
    return dict(s_iso=s_iso.reshape(shp), s_hm=s_hm.reshape(shp), outer=outer.reshape(shp),
                s_iso_raw=s_iso_raw.reshape(shp),
                peak=peak.reshape(shp), width=(s10 - s90).reshape(shp), valid=valid.reshape(shp),
                metal=metal.reshape(shp), sweep={L: v.reshape(shp) for L, v in sweep.items()})


def block_boot_median(vals, bx, by, rng):
    ok = np.isfinite(vals)
    vals, bx, by = vals[ok], bx[ok], by[ok]
    if len(vals) < 10:
        return [None, None]
    key = bx * 100000 + by
    ub, inv = np.unique(key, return_inverse=True)
    groups = [vals[inv == k] for k in range(len(ub))]
    meds = np.empty(NBOOT)
    for b in range(NBOOT):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[p] for p in pick]))
    return [rd(np.percentile(meds, 2.5)), rd(np.percentile(meds, 97.5))]


def q(v):
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": rd(np.median(v)), "mean": rd(v.mean()),
            "p25": rd(np.percentile(v, 25)), "p75": rd(np.percentile(v, 75))}


def despiked(H):
    return geom.despike(H, 2.0, 5)[0]


def window_net(Hpre, Hpost, bg):
    t = Hpost - Hpre - bg
    t = t[np.isfinite(t)]
    return float(t[t > 0].sum() * STEP * STEP), int(len(t))


def replica(A, R, roi_key="s_iso", ctrl_key="s_iso", post_level=None, roi_keep=None,
            ctrl_keep=None):
    def H(name, tp, key, keep, level=None):
        f = A[name][tp]
        h = f["sweep"][level] if (level is not None and tp == "post") else f[key]
        h = h.copy()
        if keep is not None:
            h[~keep[name]] = np.nan
        return despiked(h)
    meds = {}
    for name, reg in R.items():
        if reg["kind"] != "control":
            continue
        hp = H(name, "pre", ctrl_key, ctrl_keep)
        hq = H(name, "post", ctrl_key, ctrl_keep, post_level)
        t = (hq - hp)[np.isfinite(hq - hp)]
        meds[name] = {"median_mm": rd(np.median(t)) if len(t) else None, "cells": int(len(t)),
                      "used_for_background": bool(reg["frozen_valid"])}
    used = [v["median_mm"] for k, v in meds.items() if v["used_for_background"] and v["median_mm"] is not None]
    bg = float(np.mean(used)) if used else 0.0
    name = "roi_graft_NOT_stable"
    hp = H(name, "pre", roi_key, roi_keep)
    hq = H(name, "post", roi_key, roi_keep, post_level)
    gross, cells = window_net(hp, hq, 0.0)
    net, _ = window_net(hp, hq, bg)
    return {"controls": meds, "background_mm": rd(bg, 4), "gross_mm3": rd(gross, 1),
            "net_mm3": rd(net, 1), "roi_cells": cells, "roi_coverage": rd(cells / hp.size, 3)}


def run(case):
    t0 = time.time()
    out_case = Path(f"analysis/{case}/cross_scanner_grey_{TAG}.json")
    import os
    if os.environ.get("CSG_DRYRUN_DIR"):
        out_case = Path(os.environ["CSG_DRYRUN_DIR"]) / f"{case}_dryrun.json"
    if out_case.exists():
        raise SystemExit(f"{out_case} exists; not overwriting")
    res = json.loads(Path(f"FINAL_ARCH_20260907/results/{case}_A.json").read_text())
    cfg = json.loads(Path(f"cases/{case}.json").read_text())
    frame = geom.LocalFrame.from_dict(res["frame"])
    R = regions(res, cfg)
    paths = {"pre": Path(f"analysis/{case}/pre_cal.nii.gz"),
             "post": Path(f"cbct_blinded/{case}/post_reg.nii.gz")}
    vx = float(res["controls"]["registration_accuracy"]["voxel_mm"])
    A = {name: {} for name in R}
    grids = {}
    crops = {}
    for tp in ("pre", "post"):
        samp, cr = sample_scan(paths[tp], frame, R, vx)
        crops[tp] = cr
        for name, reg in R.items():
            s_lim = reg["s_lim_z"] if reg["sign"] > 0 else -reg["s_lim_z"]
            A[name][tp] = analyse_scan(samp[name], s_lim)
            A[name][tp]["outside_array_frac"] = samp[name]["outside_array_frac"]
            grids[name] = (samp[name]["xc"], samp[name]["yc"])
        del samp
        print(f"[{case}] {tp} sampled + analysed   peak RSS {rss_gb():.2f} GB", flush=True)

    rng = np.random.default_rng(SEED)
    structs = {}
    pooled = {"d_pre": [], "d_post": [], "delta": []}
    for name, reg in R.items():
        pre, post = A[name]["pre"], A[name]["post"]
        both = pre["valid"] & post["valid"]
        dpre = np.where(pre["valid"], pre["s_iso"] - pre["s_hm"], np.nan)
        dpost = np.where(post["valid"], post["s_iso"] - post["s_hm"], np.nan)
        delta = np.where(both, dpost - dpre, np.nan)
        xc, yc = grids[name]
        XX, YY = np.meshgrid(xc, yc)
        bx = np.floor((XX - xc[0]) / 1.0).astype(int)
        by = np.floor((YY - yc[0]) / 1.0).astype(int)
        e = {"kind": reg["kind"], "stable": reg["stable"],
             "local_box": {"X_mm": [rd(v, 2) for v in reg["X"]], "Y_mm": [rd(v, 2) for v in reg["Y"]],
                           "Z_scan_mm": [rd(v, 2) for v in reg["z"]],
                           "outward": "+Z (buccal)" if reg["sign"] > 0 else "-Z (palatal)"},
             "cells": int(both.size),
             "outside_array_sample_frac": {"pre": pre["outside_array_frac"],
                                           "post": post["outside_array_frac"]},
             "valid_pre": int(pre["valid"].sum()), "valid_post": int(post["valid"].sum()),
             "valid_both": int(both.sum()),
             "metal_flagged_pre": int(pre["metal"].sum()), "metal_flagged_post": int(post["metal"].sum()),
             "d_pre_iso_minus_halfmax_mm": q(dpre[both]),
             "d_post_iso_minus_halfmax_mm": q(dpost[both]),
             "delta_d_post_minus_pre_mm": q(delta[both]),
             "delta_d_median_CI95_block_bootstrap": block_boot_median(delta, bx, by, rng),
             "outer_plateau_grey": {"pre": q(pre["outer"][both]), "post": q(post["outer"][both])},
             "edge_peak_grey": {"pre": q(pre["peak"][both]), "post": q(post["peak"][both])},
             "halfmax_level_grey": {"pre": q(0.5 * (pre["outer"] + pre["peak"])[both]),
                                    "post": q(0.5 * (post["outer"] + post["peak"])[both])},
             "edge_width_10_90_mm": {"pre": q(pre["width"][both]), "post": q(post["width"][both])},
             "outer_plateau_airlike_frac": {
                 "pre": rd(float((pre["outer"][both] < -500).mean()) if both.any() else np.nan),
                 "post": rd(float((post["outer"][both] < -500).mean()) if both.any() else np.nan)},
             "shift_iso450_post_minus_pre_outward_mm": q((post["s_iso"] - pre["s_iso"])[both]),
             "shift_halfmax_post_minus_pre_outward_mm": q((post["s_hm"] - pre["s_hm"])[both])}
        if reg["kind"] == "control":
            e["frozen_control"] = {"valid_for_background": reg["frozen_valid"],
                                   "frozen_buccal_median_thickness_mm": reg["frozen_median_mm"]}
        structs[name] = e
        if reg["stable"]:
            pooled["d_pre"].append(dpre[both]); pooled["d_post"].append(dpost[both])
            pooled["delta"].append(delta[both])
    pooled_d = np.concatenate(pooled["delta"])
    out = {"case": case, "reader": "A", "voxel_mm": vx,
           "inputs": {k: str(v) for k, v in paths.items()},
           "results_json": f"FINAL_ARCH_20260907/results/{case}_A.json",
           "calibration_from_results": {
               "pre": {k: res["calibration"]["pre"][k] for k in ("air_grey", "soft_tissue_grey", "scale", "histogram_peaks", "native_equivalent_of_iso450")},
               "post": {k: res["calibration"]["post"][k] for k in ("air_grey", "soft_tissue_grey", "scale", "histogram_peaks", "native_equivalent_of_iso450")},
               "post_over_pre_scale_ratio": res["calibration"]["post_over_pre_scale_ratio"]},
           "crop_extents_mm": crops, "structures": structs,
           "pooled_stable": {"d_pre": q(np.concatenate(pooled["d_pre"])),
                             "d_post": q(np.concatenate(pooled["d_post"])),
                             "delta_d": q(pooled_d),
                             "members": [n for n, r_ in R.items() if r_["stable"]]}}

    area = float(res["volume"]["projected_area_mm2"])
    frozen = {"PRIMARY_net_bone_gain_mm3": res["volume"]["PRIMARY_net_bone_gain_mm3"],
              "gain_before_background_and_implant_mm3": res["volume"]["gain_before_background_and_implant_mm3"],
              "background_subtracted_mm": res["volume"]["background_subtracted_mm"],
              "projected_area_mm2": area}
    ve = {"frozen": frozen}
    ctrl_valid = [n for n, r_ in R.items() if r_["kind"] == "control" and r_["frozen_valid"]]
    dctrl = np.concatenate([np.where(A[n]["pre"]["valid"] & A[n]["post"]["valid"],
                                     (A[n]["post"]["s_iso"] - A[n]["post"]["s_hm"])
                                     - (A[n]["pre"]["s_iso"] - A[n]["pre"]["s_hm"]), np.nan).ravel()
                            for n in ctrl_valid])
    dctrl_med = float(np.nanmedian(dctrl))
    ve["a_uniform_if_uncorrected_mm3"] = {
        n: rd(structs[n]["delta_d_post_minus_pre_mm"].get("median", np.nan) * area, 1)
        for n in R if R[n]["stable"]}
    ve["a_uniform_if_uncorrected_mm3"]["pooled_stable"] = rd(np.median(pooled_d) * area, 1)
    ve["a_residual_after_control_background_mm3"] = {
        "definition": "",
        "control_plate_delta_d_median_mm": rd(dctrl_med),
        **{n: rd((structs[n]["delta_d_post_minus_pre_mm"].get("median", np.nan) - dctrl_med) * area, 1)
           for n in R if R[n]["stable"] and R[n]["kind"] != "control"}}
    rep_iso = replica(A, R)
    ve["b_replica_iso450_all_cells"] = rep_iso
    ve["b_replica_fidelity"] = {
        "gross_replica_minus_frozen_mm3": rd(rep_iso["gross_mm3"] - frozen["gain_before_background_and_implant_mm3"], 1),
        "net_replica_minus_frozen_mm3": rd(rep_iso["net_mm3"] - frozen["PRIMARY_net_bone_gain_mm3"], 1),
        "background_replica_minus_frozen_mm": rd(rep_iso["background_mm"] - frozen["background_subtracted_mm"], 3),
        "note": ""}
    common = {n: A[n]["pre"]["valid"] & A[n]["post"]["valid"] for n in R}
    nometal = {n: ~(A[n]["pre"]["metal"] | A[n]["post"]["metal"]) for n in R}
    rc_iso = replica(A, R, roi_keep=common, ctrl_keep=common)
    rc_hm = replica(A, R, roi_key="s_hm", ctrl_key="s_hm", roi_keep=common, ctrl_keep=common)

    def brief(x):
        return {k: x[k] for k in ("background_mm", "gross_mm3", "net_mm3", "roi_cells", "roi_coverage")} | {
            "control_medians_mm": {k: v["median_mm"] for k, v in x["controls"].items()}}
    variants = {
        "V0_iso450_all_cells_replica": brief(rep_iso),
        "V1_controls_exclude_cells_with_raw_gt2800_in_either_scan": brief(
            replica(A, R, ctrl_keep=nometal)),
        "V2_controls_no_titanium_punchout_field": brief(replica(A, R, ctrl_key="s_iso_raw")),
        "V2b_roi_and_controls_no_punchout_PINS_COUNTED_AS_BONE": brief(
            replica(A, R, roi_key="s_iso_raw", ctrl_key="s_iso_raw")),
        "V3_common_cells_iso450": brief(rc_iso),
        "V4_common_cells_halfmax": brief(rc_hm),
        "V5_common_roi_halfmax_controls_iso450": brief(
            replica(A, R, roi_key="s_hm", roi_keep=common, ctrl_keep=common)),
        "V6_common_roi_iso450_controls_halfmax": brief(
            replica(A, R, ctrl_key="s_hm", roi_keep=common, ctrl_keep=common)),
        "V7_roi_iso450_all_cells_controls_halfmax_valid_cells": brief(
            replica(A, R, ctrl_key="s_hm", ctrl_keep=common)),
        "V8_roi_halfmax_valid_cells_controls_iso450_all_cells": brief(
            replica(A, R, roi_key="s_hm", roi_keep=common)),
    }
    ve["b_decomposition_variants"] = variants
    ve["b_decomposition_notes"] = (
        "")
    ve["b_common_cells"] = {
        "iso450_net_mm3": rc_iso["net_mm3"], "halfmax_net_mm3": rc_hm["net_mm3"],
        "iso450_gross_mm3": rc_iso["gross_mm3"], "halfmax_gross_mm3": rc_hm["gross_mm3"],
        "iso450_background_mm": rc_iso["background_mm"], "halfmax_background_mm": rc_hm["background_mm"],
        "roi_cells": rc_iso["roi_cells"], "roi_coverage": rc_iso["roi_coverage"],
        "iso_minus_halfmax_net_mm3": rd(rc_iso["net_mm3"] - rc_hm["net_mm3"], 1),
        "iso_minus_halfmax_net_scaled_to_frozen_area_mm3": rd(
            (rc_iso["net_mm3"] - rc_hm["net_mm3"]) * area / (rc_iso["roi_cells"] * STEP * STEP), 1),
        "iso_minus_halfmax_percent_of_iso": rd(100 * (rc_iso["net_mm3"] - rc_hm["net_mm3"]) / rc_iso["net_mm3"], 1),
        "controls_iso": rc_iso["controls"], "controls_halfmax": rc_hm["controls"],
        "meaning": ""}
    sweep = []
    for L in LEVELS:
        dl = np.concatenate([np.where(common[n], (A[n]["post"]["sweep"][L] - A[n]["post"]["s_hm"])
                                      - (A[n]["pre"]["s_iso"] - A[n]["pre"]["s_hm"]), np.nan).ravel()
                             for n in ctrl_valid])
        ds = np.concatenate([np.where(common[n], (A[n]["post"]["sweep"][L] - A[n]["post"]["s_hm"])
                                      - (A[n]["pre"]["s_iso"] - A[n]["pre"]["s_hm"]), np.nan).ravel()
                             for n in R if R[n]["stable"]])
        rp = replica(A, R, post_level=L)
        sweep.append({"post_level": L, "ctrl_plate_delta_d_median_mm": rd(np.nanmedian(dl)),
                      "pooled_stable_delta_d_median_mm": rd(np.nanmedian(ds)),
                      "net_mm3": rp["net_mm3"], "gross_mm3": rp["gross_mm3"],
                      "background_mm": rp["background_mm"]})

    def solve(key):
        xs = np.array([p["post_level"] for p in sweep]); ys = np.array([p[key] for p in sweep], float)
        ns = np.array([p["net_mm3"] for p in sweep], float)
        for i in range(len(xs) - 1):
            if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]) and (ys[i] >= 0 >= ys[i + 1] or ys[i] <= 0 <= ys[i + 1]) and ys[i] != ys[i + 1]:
                t = ys[i] / (ys[i] - ys[i + 1])
                return {"L_star": rd(xs[i] + t * (xs[i + 1] - xs[i]), 1),
                        "net_at_L_star_mm3": rd(ns[i] + t * (ns[i + 1] - ns[i]), 1)}
        return {"L_star": None, "net_at_L_star_mm3": None, "note": ""}
    net450 = [p["net_mm3"] for p in sweep if p["post_level"] == ISO][0]
    ve["c_level_sweep"] = sweep
    ve["c_matched_on_control_plate"] = {**solve("ctrl_plate_delta_d_median_mm"), "net_at_450_mm3": net450}
    ve["c_matched_on_pooled_stable"] = {**solve("pooled_stable_delta_d_median_mm"), "net_at_450_mm3": net450}
    for k in ("c_matched_on_control_plate", "c_matched_on_pooled_stable"):
        m = ve[k]
        m["net_450_minus_matched_mm3"] = (rd(net450 - m["net_at_L_star_mm3"], 1)
                                          if m["net_at_L_star_mm3"] is not None else None)
    out["volume_effect"] = ve
    out["peak_rss_GiB"] = rd(rss_gb(), 2)
    out["runtime_s"] = rd(time.time() - t0, 1)
    out["numpy"] = np.__version__
    out_case.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("pooled_stable", "peak_rss_GiB")}, indent=1))
    print(json.dumps({n: (s["valid_both"], s["delta_d_post_minus_pre_mm"].get("median"))
                      for n, s in structs.items()}, indent=1))
    print(json.dumps({k: v for k, v in ve.items() if k != "c_level_sweep"}, indent=1)[:4000])


def collect():
    if OUT_FINAL.exists():
        raise SystemExit(f"{OUT_FINAL} exists; not overwriting")
    cases = {}
    for c in ("S4", "S2"):
        cases[c] = json.loads(Path(f"analysis/{c}/cross_scanner_grey_{TAG}.json").read_text())
    table = []
    for c, d in cases.items():
        for n, s in d["structures"].items():
            table.append({"case": c, "structure": n, "stable": s["stable"], "n_cells": s["valid_both"],
                          "d_pre_median_mm": s["d_pre_iso_minus_halfmax_mm"].get("median"),
                          "d_post_median_mm": s["d_post_iso_minus_halfmax_mm"].get("median"),
                          "delta_d_median_mm": s["delta_d_post_minus_pre_mm"].get("median"),
                          "delta_d_CI95": s["delta_d_median_CI95_block_bootstrap"],
                          "halfmax_level_pre": s["halfmax_level_grey"]["pre"].get("median"),
                          "halfmax_level_post": s["halfmax_level_grey"]["post"].get("median"),
                          "edge_width_pre_mm": s["edge_width_10_90_mm"]["pre"].get("median"),
                          "edge_width_post_mm": s["edge_width_10_90_mm"]["post"].get("median")})
    summary = {}
    for c, d in cases.items():
        ve = d["volume_effect"]
        summary[c] = {"pooled_stable_delta_d_median_mm": d["pooled_stable"]["delta_d"].get("median"),
                      "pooled_stable_n": d["pooled_stable"]["delta_d"].get("n"),
                      "uniform_uncorrected_mm3": ve["a_uniform_if_uncorrected_mm3"]["pooled_stable"],
                      "replica_iso450_net_mm3": ve["b_replica_iso450_all_cells"]["net_mm3"],
                      "frozen_net_mm3": ve["frozen"]["PRIMARY_net_bone_gain_mm3"],
                      "common_cells_iso450_net_mm3": ve["b_common_cells"]["iso450_net_mm3"],
                      "common_cells_halfmax_net_mm3": ve["b_common_cells"]["halfmax_net_mm3"],
                      "iso_minus_halfmax_mm3": ve["b_common_cells"]["iso_minus_halfmax_net_mm3"],
                      "iso_minus_halfmax_percent": ve["b_common_cells"]["iso_minus_halfmax_percent_of_iso"],
                      "decomposition_net_mm3": {k: v["net_mm3"] for k, v in ve["b_decomposition_variants"].items()},
                      "decomposition_background_mm": {k: v["background_mm"] for k, v in ve["b_decomposition_variants"].items()},
                      "metal_flagged_cells_pre_post": {n: [s["metal_flagged_pre"], s["metal_flagged_post"]]
                                                       for n, s in d["structures"].items()},
                      "matched_level_control_plate": ve["c_matched_on_control_plate"],
                      "matched_level_pooled_stable": ve["c_matched_on_pooled_stable"],
                      "post_over_pre_scale_ratio": d["calibration_from_results"]["post_over_pre_scale_ratio"]}
    out = {"_what": "",
           "_script": "analysis/cross_scanner_grey_reverify_2026-09-13.py",
           "_date": "", "_reader": "",
           "_sign": "",
           "summary": summary, "table": table,
           "per_case_files": {c: f"analysis/{c}/cross_scanner_grey_{TAG}.json" for c in cases}}
    OUT_FINAL.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "--collect":
        collect()
    elif arg in ("S2", "S4"):
        run(arg)
    else:
        raise SystemExit("usage: S2 | S4 | --collect")

