#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfx_common import (ROOT, SFX, READERS, RULES_SHA, PIN_APPLY1, PIN_RULE3, PROM, geom, V,
                        load_module, results, post_path, map_frozen_to_new, rss_gib, sha256_file, write_new)
from pipeline import landmarks

GRID = 0.2
TOL_GATE = 0.90
ELIGIBLE = {"S1": [], "S2": [63, 64], "S3": [70], "S4": [441, 735, 760]}
D_MARGINS = (1.0, 1.4, 2.0)
D_PRIMARY = 1.4
TIE = 1e-9
SPEC_FRACS = (0.15, 0.20, 0.30)
SPEC_GAPS = (2.0, 4.0)
BIG_X = (-25.0, 55.0)
BIG_DEPTH = 20.0
WIN_X = (10.0, 20.0)
WIN_Y0, WIN_D = 15.0, 10.0
KEEP_Y = (11.0, 42.0)
KEEP_X = (-40.0, 60.0)


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def gain_u(Hpre, H, bg):
    t = H - Hpre - bg
    t = t[np.isfinite(t)]
    return float(t[t > 0].sum() * GRID * GRID)


class Ctx:
    pass


def setup(case, post, depth_json):
    c = Ctx()
    c.case, c.post = case, post
    c.A1 = load_module("sfx_pin_apply1", PIN_APPLY1)
    c.RULE1, c.rule1, _ = c.A1.load_rule()
    c.RULE3 = load_module("sfx_pin_rule3", PIN_RULE3)
    c.PROMm = load_module("sfx_prom", PROM)
    au = json.loads((ROOT / f"analysis/{case}/fixation_pins_2026-09-13.json").read_text())
    c.lo = np.array(au["crop_lo_vox"]); c.hi = np.array(au["crop_hi_vox"])
    c.sl = tuple(slice(int(a), int(b)) for a, b in zip(c.lo, c.hi))
    c.res = {R: results(case, R) for R in READERS}
    c.frames = {R: geom.LocalFrame.from_dict(c.res[R]["frame"]) for R in READERS}
    c.bands = {R: [float(v) for v in c.res[R]["search_band_Z_mm"]] for R in READERS}
    c.excluded = set(json.loads((ROOT / f"cases/{case}.json").read_text()).get("exclude_control_windows") or [])
    cont = json.loads(Path(depth_json).read_text())["cases"][case]
    depth = 10.0
    if cont["contiguous_from_window_edge"]:
        for lo_, hi_ in sorted(tuple(float(x) for x in b.split("-")) for b in cont["bands_above_null"]):
            if lo_ == depth:
                depth = hi_
    c.depth = depth
    c.depth_json = str(depth_json)
    c.pre_c, c.vx = c.RULE1.read_crop(ROOT / f"analysis/{case}/pre_cal.nii.gz", c.sl)
    c.post_c, vx2 = c.RULE1.read_crop(post_path(case, post), c.sl)
    assert abs(c.vx - vx2) < 1e-6
    c.origin_mm = c.lo * c.vx
    if post == "frozen":
        lbl, pins, imp_label, rows = c.RULE1.pin_components(case, c.post_c, c.pre_c, c.lo, c.vx)
        c.pin_vox = {L: np.argwhere(lbl == L) + c.lo for L in ELIGIBLE[case]}
        c.implant_idx = (np.argwhere(lbl == imp_label) + c.lo) if imp_label is not None else None
        c.titanium_match = {"method": ""}
        del lbl
    else:
        frz_c, _ = c.RULE1.read_crop(post_path(case, "frozen"), c.sl)
        lblF, _, impF, _ = c.RULE1.pin_components(case, frz_c, c.pre_c, c.lo, c.vx)
        lbl, n = ndi.label(c.post_c > geom.METAL_LEVEL)
        cnt = np.bincount(lbl.ravel())
        c.pin_vox, c.titanium_match = {}, {"method": "", "objects": {}}
        objs = {L: np.argwhere(lblF == L) + c.lo for L in ELIGIBLE[case]}
        if impF is not None:
            objs["implant"] = np.argwhere(lblF == impF) + c.lo
        for nm, vox in objs.items():
            cen = vox.mean(0) * c.vx
            m = map_frozen_to_new(case, post, cen[None])[0]
            mi = np.round(m / c.vx).astype(int) - c.lo
            rad = int(np.ceil(2.0 / c.vx))
            sub = lbl[tuple(slice(max(a - rad, 0), a + rad + 1) for a in mi)]
            cand = [k for k in np.unique(sub) if k > 0]
            best, bd = None, 1e9
            for k in cand:
                idx = np.argwhere(lbl == k)
                d = np.linalg.norm((idx.mean(0) + c.lo) * c.vx - m)
                if d < bd:
                    best, bd = k, d
            assert best is not None and bd < 1.0, (nm, bd)
            nv = int(cnt[best])
            assert 0.8 < nv / len(vox) < 1.25, (nm, nv, len(vox))
            newvox = np.argwhere(lbl == best) + c.lo
            c.titanium_match["objects"][str(nm)] = {"frozen_voxels": int(len(vox)), "new_voxels": nv,
                                                    "mapped_centroid_to_new_centroid_mm": round(float(bd), 4),
                                                    "frozen_to_new_centroid_shift_mm": round(float(np.linalg.norm(
                                                        newvox.mean(0) * c.vx - cen)), 4)}
            if nm == "implant":
                c.implant_idx = newvox
            else:
                c.pin_vox[nm] = newvox
        if impF is None:
            c.implant_idx = None
        del lbl, lblF, frz_c
    c.elig = list(ELIGIBLE[case])
    c.pin_idx = np.vstack([c.pin_vox[L] for L in c.elig]) if c.elig else None
    pf = geom.bone_field(c.pre_c)
    Vp, Fp = geom.iso_surface(pf, c.vx)
    del pf
    Vp = Vp + c.origin_mm
    qf = geom.bone_field(c.post_c)
    Vq, Fq = geom.iso_surface(qf, c.vx)
    del qf
    Vq = Vq + c.origin_mm
    c.KEEP, c.P, c.Q = {}, {}, {}
    for R in READERS:
        z0, z1 = c.bands[R]
        c.KEEP[R] = geom.window_box({"X_mm": list(KEEP_X), "Y_mm": list(KEEP_Y)}, {}, z0, z1, pad_x_mm=0.0, pad_y_mm=0.0)
        c.P[R] = geom.surface_points_local(Vp, Fp, c.frames[R], c.KEEP[R])
        c.Q[R] = geom.surface_points_local(Vq, Fq, c.frames[R], c.KEEP[R])
    c.surfaces = (Vp, Fp, Vq, Fq)
    pre_full, vxf = geom.load_volume(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"))
    post_full, _ = geom.load_volume(str(post_path(case, post)))
    pff, qff = geom.bone_field(pre_full), geom.bone_field(post_full)
    del pre_full, post_full
    c.md_prof = {}
    for R in READERS:
        z0, z1 = c.bands[R]
        c.md_prof[R] = V.new_bone_md_profile(pff, qff, c.frames[R], vxf, z0, z1)
    del pff, qff
    return c


def world_S(frame, X, Y, Z):
    return frame.to_anatomical(np.column_stack([X, Y, Z]))[:, 0]


def window_height(frame, Hpre, xe, ye):
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    XC, YC = np.meshgrid(xc, yc)
    ok = np.isfinite(Hpre)
    if not ok.any():
        return None
    return float(np.median(world_S(frame, XC[ok], YC[ok], Hpre[ok])))


def window_height_range(frame, Hpre, xe, ye):
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    XC, YC = np.meshgrid(xc, yc)
    ok = np.isfinite(Hpre)
    S = world_S(frame, XC[ok], YC[ok], Hpre[ok])
    return float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))


def filter_by_height(frame, m, S_range, keep=None):
    Hpre, th, xe, ye = m["_Hpre"], m["_thickness"], m["_x_edges"], m["_y_edges"]
    if keep is None:
        xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
        XC, YC = np.meshgrid(xc, yc)
        S = np.full(Hpre.shape, np.nan)
        ok = np.isfinite(Hpre)
        S[ok] = world_S(frame, XC[ok], YC[ok], Hpre[ok])
        keep = ok & (S >= S_range[0]) & (S <= S_range[1])
    t = np.where(keep, th, np.nan)
    okt = np.isfinite(t)
    if not okt.any():
        return {"cells": 0, "_keep": keep}
    tt = t[okt]
    return {"cells": int(okt.sum()), "coverage": round(float(okt.mean()), 3),
            "median_thickness_mm": round(float(np.median(tt)), 3), "mean_thickness_mm": round(float(tt.mean()), 3),
            "_keep": keep, "height_range_S_mm": [round(S_range[0], 2), round(S_range[1], 2)] if S_range else None}


def snap(v):
    return round(round(v / GRID) * GRID, 10)


def height_band(c, R, xr, yband_win, S_win):
    frame = c.frames[R]; z0, z1 = c.bands[R]
    eS = frame.R[:, 0]
    xm = 0.5 * (xr[0] + xr[1]); xw = 0.5 * (WIN_X[0] + WIN_X[1])
    dY = snap(-eS[0] * (xm - xw) / eS[1])
    hist = [{"iteration": 0, "dY_mm": dY, "source": "frame prediction"}]
    S_c = None
    for it in (1, 2):
        yb = [yband_win[0] + dY, yband_win[1] + dY]
        xe, ye = V._edges(*xr, GRID), V._edges(*yb, GRID)
        Hp = geom.height_field(c.P[R], xe, ye, z0, z1, mode="top_mean", spike_tol_mm=2.0)
        S_c = window_height(frame, Hp, xe, ye)
        if S_c is None:
            break
        dY = snap(dY + (S_win - S_c) / eS[1])
        hist.append({"iteration": it, "S_control_median_mm": round(S_c, 3), "dY_next_mm": dY})
    yb = [yband_win[0] + dY, yband_win[1] + dY]
    xe, ye = V._edges(*xr, GRID), V._edges(*yb, GRID)
    Hp = geom.height_field(c.P[R], xe, ye, z0, z1, mode="top_mean", spike_tol_mm=2.0)
    S_final = window_height(frame, Hp, xe, ye)
    return [round(yb[0], 10), round(yb[1], 10)], {"dY_mm": dY, "S_window_median_mm": round(S_win, 3),
                                                  "S_control_median_mm": r(S_final, 3),
                                                  "residual_height_mismatch_mm": r(S_final - S_win, 3) if S_final is not None else None,
                                                  "iterations": hist}


def measure_controls(c, R, ctrl, rule, yband_win, Hpre_win, xe_w, ye_w, palatal=True, min_cells=200,
                     drop_over_tol=False):
    frame = c.frames[R]; z0, z1 = c.bands[R]; KEEP = c.KEEP[R]
    S_win = window_height(frame, Hpre_win, xe_w, ye_w) if rule == "height" else None
    S_range = window_height_range(frame, Hpre_win, xe_w, ye_w) if rule == "heightfilter" else None
    out, bg_vals, ling_vals, abort = {}, [], [], []
    for nm, xr in ctrl.items():
        if nm in c.excluded:
            out[nm] = {"X_mm": xr, "excluded_by_config": True, "valid": False}
            continue
        if rule == "height":
            yb, hrec = height_band(c, R, xr, yband_win, S_win)
        else:
            yb, hrec = list(yband_win), None
        b = V.measure_window(c.P[R], c.Q[R], xr, yb, GRID, z0, z1, keep_box=KEEP)
        if rule == "heightfilter" and b.get("cells", 0):
            b = filter_by_height(frame, b, S_range)
        rec = {"X_mm": [float(v) for v in xr], "Y_mm": yb, "height_match": hrec, "cells": b.get("cells", 0)}
        if b.get("cells", 0) < min_cells:
            rec.update(valid=False, reason=f"only {b.get('cells', 0)} cells")
            out[nm] = rec
            continue
        bo = b["median_thickness_mm"]
        rec.update(buccal_median_mm=bo, buccal_mean_mm=b["mean_thickness_mm"], coverage=b["coverage"])
        if palatal:
            l = V.measure_window(c.P[R], c.Q[R], xr, yb, GRID, z0, z1, mode="bottom_mean", keep_box=KEEP)
            if rule == "heightfilter" and l.get("cells", 0):
                l = filter_by_height(frame, l, S_range, keep=b.get("_keep"))
            lo_ = -l.get("median_thickness_mm", 0.0) if l.get("cells", 0) >= 200 else None
            rec["lingual_outward_mm"] = None if lo_ is None else round(lo_, 3)
            if lo_ is not None:
                sh, di = (bo - lo_) / 2.0, (bo + lo_) / 2.0
                rec["rigid_shift_mm"], rec["dilation_mm"] = round(sh, 3), round(di, 3)
                if not (abs(sh) <= TOL_GATE and abs(di) <= TOL_GATE):
                    rec.update(valid=False, reason="palatal-witness test failed")
                    out[nm] = rec
                    continue
                ling_vals.append(lo_)
            if abs(bo) > TOL_GATE:
                abort.append(f"{nm} buccal median {bo:+.3f} mm exceeds {TOL_GATE}")
        elif drop_over_tol and abs(bo) > TOL_GATE:
            rec.update(valid=False, reason="over 0.90 mm (dropped, sweep convention)")
            out[nm] = rec
            continue
        rec["valid"] = True
        out[nm] = rec
        bg_vals.append(bo)
    if len(bg_vals) > 1 and (max(bg_vals) - min(bg_vals)) > 2 * TOL_GATE:
        abort.append(f"controls disagree by {max(bg_vals) - min(bg_vals):.3f} mm")
    bg = float(np.mean(bg_vals)) if bg_vals else None
    lingual = float(np.mean(ling_vals)) if ling_vals else None
    acc = None
    if bg is not None and lingual is not None:
        acc = {"rigid_shift_mm": round((bg - lingual) / 2.0, 3), "surface_dilation_mm": round((bg + lingual) / 2.0, 3)}
    return {"windows": out, "background_unrounded": bg, "background_mm": r(bg, 3), "lingual_unrounded": lingual,
            "registration_accuracy": acc, "abort_conditions": abort}


def d7e_surface(c, R, xe, ye, Hpre, Hpost, prom_flag, d):
    frame = c.frames[R]
    ny, nx = Hpost.shape
    if not c.elig:
        z = np.zeros((ny, nx), bool)
        return Hpost.copy(), z, z.copy(), 0
    Lall = c.frames[R].to_local(c.pin_idx.astype(float) * c.vx)
    F_all, Z = c.RULE3.zhead_field(xe, ye, Lall, d)
    Hc3, _ = c.RULE3.cap_surface(Hpre, Hpost, F_all, Z, c.vx)
    loc = {L: frame.to_local(c.pin_vox[L].astype(float) * c.vx) for L in c.elig}
    Dst = np.stack([c.PROMm.dist_field(xe, ye, loc[L][:, :2]) for L in c.elig])
    assert np.array_equal(Dst.min(0) <= d + 1e-6, F_all)
    order = np.argsort(Dst, axis=0, kind="stable")
    i0 = order[0]
    pflag = np.array([prom_flag[L] for L in c.elig])
    ties = np.zeros((ny, nx), bool)
    if len(c.elig) > 1:
        d0 = np.take_along_axis(Dst, order[:1], 0)[0]
        d1 = np.take_along_axis(Dst, order[1:2], 0)[0]
        ties = F_all & (np.abs(d1 - d0) <= TIE) & (pflag[i0] != pflag[order[1]])
    Fcap = F_all & pflag[i0] & ~ties
    Hc = Hpost.copy()
    Hc[Fcap] = Hc3[Fcap]
    c.RULE3.assert_cell_properties(Hpre, Hpost, Hc, Fcap)
    c.RULE3.assert_cell_properties(Hpre, Hpost, Hc, F_all)
    return Hc, F_all, Fcap, int(ties.sum())


def prominence(c, R, Hpre, Hpost, xe, ye):
    frame = c.frames[R]
    loc = {L: frame.to_local(c.pin_vox[L].astype(float) * c.vx) for L in c.elig}
    D = {L: c.PROMm.dist_field(xe, ye, loc[L][:, :2]) for L in c.elig}
    out = {}
    for L in c.elig:
        ztop = float(loc[L][:, 2].max())
        rec = c.PROMm.prominence(Hpost, Hpre, D, L, c.elig, ztop, c.vx)
        rec["Ztop_mm"] = round(ztop, 4)
        rec["titanium_voxels"] = int(len(c.pin_vox[L]))
        out[L] = rec
    return out


def ti_term(c, R, vox, Hpre, H, xe, ye):
    if vox is None or not len(vox):
        return 0.0, 0.0
    t = V.implant_volume_in_envelope(vox, c.frames[R], Hpre, H, xe, ye, c.vx)["implant_in_envelope_mm3"]
    u = c.A1.envelope_overlap(vox, c.frames[R], Hpre, H, xe, ye, c.vx)
    assert t == round(u, 1)
    return t, u


def metal_frac_cells(frame, vox, Hpre, Hs, xe, ye, vx):
    ny, nx = Hpre.shape
    out = np.zeros((ny, nx))
    if vox is None or not len(vox):
        return out
    P = frame.to_local(np.asarray(vox, float) * vx)
    ix = np.digitize(P[:, 0], xe) - 1
    iy = np.digitize(P[:, 1], ye) - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not inside.any():
        return out
    lo = Hpre[iy[inside], ix[inside]]
    hi = Hs[iy[inside], ix[inside]]
    z = P[inside, 2]
    half = vx / 2.0
    frac = np.clip(np.minimum(hi, z + half) - np.maximum(lo, z - half), 0.0, vx) / vx
    frac = np.where(np.isfinite(lo) & np.isfinite(hi), frac, 0.0)
    np.add.at(out, (iy[inside], ix[inside]), frac)
    return out


def run_reader(c, R, ctrl_rule, npz):
    res = c.res[R]; frame = c.frames[R]; z0, z1 = c.bands[R]; KEEP = c.KEEP[R]
    roi = res["roi"]
    md = c.md_prof[R]
    ext = V.graft_extent(md)
    out = {"reader": R, "graft_extent": {k: v for k, v in ext.items() if not k.startswith("_")},
           "md_profile": {k: v for k, v in md.items() if not k.startswith("_")}}
    m_nobg = V.measure_window(c.P[R], c.Q[R], roi["X_mm"], roi["Y_mm"], GRID, z0, z1, background=0.0, keep_box=KEEP)
    Hpre, Hpost, xe, ye = m_nobg["_Hpre"], m_nobg["_Hpost"], m_nobg["_x_edges"], m_nobg["_y_edges"]
    ctrl = landmarks.control_windows_clean(roi, md, md_positive_is=frame.md_positive_is)
    placement = ctrl.pop("_placement"); ctrl.pop("_rule")
    cw = measure_controls(c, R, ctrl, ctrl_rule, roi["Y_mm"], Hpre, xe, ye)
    cw["placement"] = placement
    bg = cw["background_unrounded"]
    out["controls"] = cw
    main = V.measure_window(None, None, None, None, GRID, background=bg, fields=(Hpre, Hpost, xe, ye))
    raw = V.measure_window(None, None, None, None, GRID, background=0.0, fields=(Hpre, Hpost, xe, ye))
    imp_t, imp_u = ti_term(c, R, c.implant_idx, Hpre, Hpost, xe, ye)
    as_exec = round(main["gain_mm3"] - imp_t, 1)
    mmax = V.measure_window(c.P[R], c.Q[R], roi["X_mm"], roi["Y_mm"], GRID, z0, z1, background=0.0, mode="max",
                            spike_tol_mm=0.0, keep_box=KEEP)
    prof0 = V.gain_profile(main["_thickness"], xe, ye, roi)
    out["as_executed"] = {"net_mm3": as_exec, "gross_mm3": main["gain_mm3"],
                          "gain_before_background_mm3": raw["gain_mm3"], "implant_in_envelope_mm3": imp_t,
                          "implant_unrounded": imp_u, "background_mm": r(bg, 3), "background_unrounded": bg,
                          "loss_mm3": main["loss_mm3"], "area_mm2": main["area_mm2"], "coverage": main["coverage"],
                          "mean_thickness_mm": main["mean_thickness_mm"], "max_thickness_mm": main["max_thickness_mm"],
                          "volume_to_surface_ratio": round(as_exec / main["area_mm2"], 3),
                          "python_cloudcompare_equivalent_mm3": mmax["gain_mm3"],
                          "linear_gain_levels": prof0["levels_mm_apical_to_platform"],
                          "linear_gain_peak": prof0["peak"], "linear_gain_overall_mean_mm": prof0["overall_mean_mm"]}
    prom = prominence(c, R, Hpre, Hpost, xe, ye) if c.elig else {}
    prom_flag = {L: bool(prom[L]["cap_applied"]) for L in c.elig}
    out["prominence"] = {str(L): {k: v for k, v in p.items() if k in ("P_mm", "decision", "cap_applied", "Ztop_mm",
                                                                       "titanium_voxels", "core_0-0.5mm", "ring_1.4-3.0mm")}
                         for L, p in prom.items()}
    d7 = {}
    for d in D_MARGINS:
        Hc, F_all, Fcap, ties = d7e_surface(c, R, xe, ye, Hpre, Hpost, prom_flag, d)
        m = V.measure_window(None, None, None, None, GRID, background=bg, fields=(Hpre, Hc, xe, ye))
        pin_t, pin_u = ti_term(c, R, c.pin_idx, Hpre, Hc, xe, ye)
        net = round(m["gain_mm3"] - imp_t - pin_t, 1)
        g0 = gain_u(Hpre, Hpost, bg); g1 = gain_u(Hpre, Hc, bg)
        d7[f"d={d:.1f}"] = {"net_mm3": net, "gross_mm3": m["gain_mm3"], "metal_term_mm3": pin_t, "metal_unrounded": pin_u,
                            "cap_term_unrounded_mm3": round(g1 - g0, 4), "capped_cells": int(Fcap.sum()),
                            "footprint_cells": int(F_all.sum()), "cells_lowered": int((np.isfinite(Hpost) & (Hc < Hpost)).sum()),
                            "ties": ties, "loss_mm3": m["loss_mm3"], "area_mm2": m["area_mm2"], "coverage": m["coverage"]}
        npz[f"{R}_primary_10x10_d{d:.1f}_Hc"] = Hc
        if d == D_PRIMARY:
            Hc_p, pin_p, pin_pu = Hc, pin_t, pin_u
            prof = V.gain_profile(m["_thickness"], xe, ye, roi)
            depth = prof["_depth"]
            lev = {}
            for k in ("1", "2", "3", "5", "7"):
                i = int(np.argmin(np.abs(depth - float(k))))
                lev[k] = {"mean_mm": prof["levels_mm_apical_to_platform"][k][0],
                          "max_mm": prof["levels_mm_apical_to_platform"][k][1],
                          "finite_cells_in_row": int(np.isfinite(m["_thickness"][i]).sum()),
                          "cells_in_row": int(m["_thickness"][i].size)}
            primary = {"net_mm3": net, "gross_mm3": m["gain_mm3"], "implant_mm3": imp_t, "pin_metal_mm3": pin_t,
                       "cap_term_unrounded_mm3": round(g1 - g0, 4),
                       "area_mm2": m["area_mm2"], "coverage": m["coverage"],
                       "volume_to_surface_ratio": round(net / m["area_mm2"], 3),
                       "inward_shift_native_loss_mm3": m["loss_mm3"],
                       "linear_gain_levels": lev, "linear_gain_peak": prof["peak"],
                       "linear_gain_overall_mean_mm": prof["overall_mean_mm"]}
            if c.elig:
                Hn = Hpost
                mn = V.measure_window(None, None, None, None, GRID, background=bg, fields=(Hpre, Hn, xe, ye))
                pn, _ = ti_term(c, R, c.pin_idx, Hpre, Hn, xe, ye)
                primary["metal_only_no_cap_net_mm3"] = round(mn["gain_mm3"] - imp_t - pn, 1)
                for L in c.elig:
                    if prom_flag[L]:
                        f2 = dict(prom_flag); f2[L] = False
                        Hx, _, _, _ = d7e_surface(c, R, xe, ye, Hpre, Hpost, f2, D_PRIMARY)
                        mx = V.measure_window(None, None, None, None, GRID, background=bg, fields=(Hpre, Hx, xe, ye))
                        px, _ = ti_term(c, R, c.pin_idx, Hpre, Hx, xe, ye)
                        primary[f"without_cap_of_pin_{L}_net_mm3"] = round(mx["gain_mm3"] - imp_t - px, 1)
    primary["margins"] = {k: v["net_mm3"] for k, v in d7.items()}
    out["d7e"] = d7
    out["primary"] = primary
    npz[f"{R}_primary_10x10_Hpre"] = Hpre; npz[f"{R}_primary_10x10_Hpost"] = Hpost
    npz[f"{R}_primary_10x10_xe"] = xe; npz[f"{R}_primary_10x10_ye"] = ye
    bg4, bfz = round(bg, 4), round(bg, 3)

    def net_bg(bg_member):
        dbg = round(bg_member, 4) - bfz
        g = gain_u(Hpre, Hc_p, bg4 + dbg)
        return round(round(g, 1) - imp_t - pin_p, 1)

    mem = {}
    valid = {nm: w for nm, w in cw["windows"].items() if w.get("valid")}
    if len(valid) == 2:
        (n1, w1), (n2, w2) = valid.items()
        x1, x2 = np.mean(w1["X_mm"]), np.mean(w2["X_mm"])
        m1_, m2_ = w1["buccal_median_mm"], w2["buccal_median_mm"]
        dist = m1_ + (m2_ - m1_) * (15.0 - x1) / (x2 - x1)
        mem["DIST_WEIGHTED"] = {"background_mm": round(dist, 4), "net_mm3": net_bg(dist)}
    for nm, w in valid.items():
        mem[f"WINDOW_{nm}_alone"] = {"background_mm": w["buccal_median_mm"], "net_mm3": net_bg(w["buccal_median_mm"])}
    out["bg_members"] = mem
    specs = []
    for frac in SPEC_FRACS:
        for gap in SPEC_GAPS:
            cs = landmarks.control_windows_clean(roi, md, gap_mm=gap, frac=frac, md_positive_is=frame.md_positive_is)
            pl = cs.pop("_placement"); cs.pop("_rule")
            rec = {"frac": frac, "gap": gap, "X_mm": {k: v for k, v in cs.items()}}
            if any(not m_.get("clean") for m_ in pl.values()):
                rec["admissible"] = False; rec["reason"] = "placement not clean"; specs.append(rec); continue
            sc = measure_controls(c, R, cs, ctrl_rule, roi["Y_mm"], Hpre, xe, ye)
            ok = True
            bgs = []
            for nm, w in sc["windows"].items():
                if w.get("excluded_by_config"):
                    continue
                if w.get("cells", 0) < 200:
                    ok = False; break
                if not w.get("valid") and w.get("reason") == "palatal-witness test failed":
                    continue
                if abs(w["buccal_median_mm"]) > TOL_GATE:
                    ok = False; break
                bgs.append(w["buccal_median_mm"])
            if not ok or not bgs:
                rec["admissible"] = False; rec["reason"] = "control gate"; specs.append(rec); continue
            bgs_ = float(np.mean(bgs))
            rec.update(admissible=True, background_mm=round(bgs_, 3), n_controls=len(bgs),
                       controls={k: {kk: w.get(kk) for kk in ("Y_mm", "buccal_median_mm", "valid")} for k, w in sc["windows"].items()})
            rec["net_step0.2_T1_mm3"] = net_bg(round(bgs_, 3))
            md_ = V.measure_window(None, None, None, None, GRID, background=bgs_, fields=(Hpre, Hc_p, xe, ye))
            rec["net_step0.2_direct_mm3"] = round(md_["gain_mm3"] - imp_t - pin_p, 1)
            rec["net_step0.2_mm3"] = rec["net_step0.2_direct_mm3"] if not c.elig else rec["net_step0.2_T1_mm3"]
            for st in (0.15, 0.20, 0.30):
                ms = V.measure_window(c.P[R], c.Q[R], roi["X_mm"], roi["Y_mm"], st, z0, z1, background=bgs_, keep_box=KEEP)
                rec[f"asexec_step{st}_mm3"] = round(ms["gain_mm3"] - imp_t, 1)
                if c.case == "S1" and st != 0.20:
                    ti, _ = ti_term(c, R, c.implant_idx, ms["_Hpre"], ms["_Hpost"], ms["_x_edges"], ms["_y_edges"])
                    rec[f"net_step{st}_mm3"] = round(ms["gain_mm3"] - ti, 1)
            specs.append(rec)
    out["specs"] = specs
    if c.depth != 10.0:
        roi_s = landmarks.standard_roi(10.0, c.depth, offset=frame.offset, anchor="preop_anatomy")
        ms0 = V.measure_window(c.P[R], c.Q[R], roi_s["X_mm"], roi_s["Y_mm"], GRID, z0, z1, background=0.0, keep_box=KEEP)
        Hs_pre, Hs_post, xs, ys = ms0["_Hpre"], ms0["_Hpost"], ms0["_x_edges"], ms0["_y_edges"]
        roi_w = landmarks.standard_roi(10.0, 10.0, offset=frame.offset, anchor="preop_anatomy")
        cs = landmarks.control_windows_clean(roi_w, md, gap_mm=2.0, frac=0.20, md_positive_is=frame.md_positive_is)
        cs.pop("_placement"); cs.pop("_rule")
        sc = measure_controls(c, R, cs, ctrl_rule, roi_s["Y_mm"], Hs_pre, xs, ys, palatal=False, drop_over_tol=True)
        sbg = sc["background_unrounded"] if sc["background_unrounded"] is not None else 0.0
        msb = V.measure_window(None, None, None, None, GRID, background=sbg, fields=(Hs_pre, Hs_post, xs, ys))
        simp = imp_t * c.depth / 10.0
        sec = {"depth_mm": c.depth, "controls": sc, "background_mm": r(sbg, 4),
               "as_executed_net_mm3": round(msb["gain_mm3"] - simp, 1), "implant_term_scaled_mm3": round(simp, 3)}
        for d in D_MARGINS:
            Hc_s, _, Fc_s, _ = d7e_surface(c, R, xs, ys, Hs_pre, Hs_post, prom_flag, d)
            m_ = V.measure_window(None, None, None, None, GRID, background=sbg, fields=(Hs_pre, Hc_s, xs, ys))
            pt, _ = ti_term(c, R, c.pin_idx, Hs_pre, Hc_s, xs, ys)
            sec[f"net_d{d:.1f}_mm3"] = round(m_["gain_mm3"] - simp - pt, 1)
            npz[f"{R}_secondary_d{d:.1f}_Hc"] = Hc_s
        npz[f"{R}_secondary_Hpre"] = Hs_pre; npz[f"{R}_secondary_Hpost"] = Hs_post
        npz[f"{R}_secondary_xe"] = xs; npz[f"{R}_secondary_ye"] = ys
        out["secondary"] = sec
    out["extended"] = extended(c, R, Hpre, Hpost, Hc_p, xe, ye, bg, imp_t, imp_u, pin_p, pin_pu, prom_flag, ext,
                               primary["net_mm3"], npz)
    return out


def big_edges(xe_w, ye_w):
    nL = int(round((xe_w[0] - BIG_X[0]) / GRID))
    nR = int(round((BIG_X[1] - xe_w[-1]) / GRID))
    nA = int(round((ye_w[0] + BIG_DEPTH - ye_w[-1]) / GRID))
    xe = np.concatenate([xe_w[0] - GRID * np.arange(nL, 0, -1), xe_w, xe_w[-1] + GRID * np.arange(1, nR + 1)])
    ye = np.concatenate([ye_w, ye_w[-1] + GRID * np.arange(1, nA + 1)])
    return xe, ye, nL


def extended(c, R, Hpre_w, Hpost_w, Hc_w, xe_w, ye_w, bg, imp_t, imp_u, pin_t, pin_u, prom_flag, ext, net_primary, npz):
    frame = c.frames[R]; z0, z1 = c.bands[R]
    xe, ye, nL = big_edges(xe_w, ye_w)
    Hpre = geom.height_field(c.P[R], xe, ye, z0, z1, mode="top_mean", spike_tol_mm=2.0)
    Hpost = geom.height_field(c.Q[R], xe, ye, z0, z1, mode="top_mean", spike_tol_mm=2.0)
    Hc, _, _, _ = d7e_surface(c, R, xe, ye, Hpre, Hpost, prom_flag, D_PRIMARY)
    impc = metal_frac_cells(frame, c.implant_idx, Hpre, Hc, xe, ye, c.vx)
    pinc = metal_frac_cells(frame, c.pin_idx, Hpre, Hc, xe, ye, c.vx)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    XC, YC = np.meshgrid(xc, yc)
    depth = c.ext_depth
    E_lo, E_hi = float(ext["x_from_mm"]) - 0.5, float(ext["x_to_mm"]) + 0.5
    U_lo, U_hi = min(WIN_X[0], E_lo), max(WIN_X[1], E_hi)
    Y_hi = WIN_Y0 + depth
    in_xw = (XC > WIN_X[0]) & (XC < WIN_X[1])
    in_yw = (YC > WIN_Y0) & (YC < WIN_Y0 + WIN_D)
    beyond_y = (YC > WIN_Y0 + WIN_D) & (YC < Y_hi)
    lo_x = (XC > U_lo) & (XC < WIN_X[0]); hi_x = (XC > WIN_X[1]) & (XC < U_hi)
    masks = {"window": in_xw & in_yw, "lowX_strip": lo_x & in_yw, "highX_strip": hi_x & in_yw,
             "apical_strip": in_xw & beyond_y, "lowX_apical_corner": lo_x & beyond_y, "highX_apical_corner": hi_x & beyond_y}
    assert int(masks["window"].sum()) == 2500

    def stats(mask):
        t = Hc - Hpre - bg
        ok = mask & np.isfinite(t)
        tt = t[ok]
        gain = float(tt[tt > 0].sum() * GRID * GRID)
        imp = float(impc[mask].sum()) * c.vx ** 3
        pin = float(pinc[mask].sum()) * c.vx ** 3
        return {"gain_before_titanium_mm3": r(gain, 4), "implant_mm3": r(imp, 4), "pin_mm3": r(pin, 4),
                "net_mm3": r(gain - imp - pin, 4), "loss_mm3": r(float(-tt[tt < 0].sum() * GRID * GRID), 4),
                "measured_area_mm2": r(ok.sum() * GRID * GRID, 2), "_net": gain - imp - pin}
    reg = {nm: stats(m) for nm, m in masks.items()}
    win_u = gain_u(Hpre_w, Hc_w, bg) - imp_u - pin_u
    outside = sum(reg[nm]["_net"] for nm in masks if nm != "window")
    whole = win_u + outside
    md_pos = frame.md_positive_is
    names = ({"lowX": "distal", "highX": "mesial"} if md_pos == "mesial" else {"lowX": "mesial", "highX": "distal"})
    out = {"extent_X_mm": [E_lo, E_hi], "union_X_mm": [U_lo, U_hi], "Y_mm": [WIN_Y0, Y_hi], "depth_mm": depth,
           "extent_mm": ext["extent_mm"], "regions": {(nm.replace("lowX", names["lowX"]).replace("highX", names["highX"])):
                                                       {k: v for k, v in s.items() if not k.startswith("_")} for nm, s in reg.items()},
           "whole_net_mm3_unrounded": r(whole, 4), "whole_net_mm3": round(whole, 1),
           "window_share_pct": round(100.0 * net_primary / whole, 1) if whole > 0 else None,
           "outside_net_mm3": r(outside, 4)}
    npz[f"{R}_big_Hpre"] = Hpre; npz[f"{R}_big_Hpost"] = Hpost; npz[f"{R}_big_Hc"] = Hc
    npz[f"{R}_big_xe"] = xe; npz[f"{R}_big_ye"] = ye
    return out


def threshold_sweep(c, R, bg, levels=(350.0, 550.0, 750.0)):
    res = c.res[R]; frame = c.frames[R]; z0, z1 = c.bands[R]; roi = res["roi"]
    out = {}
    pf = geom.bone_field(c.pre_c); qf = geom.bone_field(c.post_c)
    for lv in levels:
        Vp, Fp = geom.iso_surface(pf, c.vx, level=lv); Vp += c.origin_mm
        Vq, Fq = geom.iso_surface(qf, c.vx, level=lv); Vq += c.origin_mm
        m = V.measure_window(geom.surface_points_local(Vp, Fp, frame, c.KEEP[R]),
                             geom.surface_points_local(Vq, Fq, frame, c.KEEP[R]),
                             roi["X_mm"], roi["Y_mm"], GRID, z0, z1, background=bg, keep_box=c.KEEP[R])
        out[f"{lv:.0f}"] = m["gain_mm3"]
        del Vp, Fp, Vq, Fq, m
    del pf, qf
    return out


def results_like(c, R, rr, ctrl_rule):
    res = json.loads(json.dumps(c.res[R]))
    res["_scalefix"] = {"post": c.post, "ctrl_rule": ctrl_rule, "rules_sha256": RULES_SHA,
                        "note": ""}
    res["new_bone_md_profile"] = rr["md_profile"]
    res["graft_extent_SECONDARY"] = rr["graft_extent"]
    res["control_windows"] = {nm: w["X_mm"] for nm, w in rr["controls"]["windows"].items()}
    res["control_windows_Y_mm"] = {nm: w.get("Y_mm") for nm, w in rr["controls"]["windows"].items()}
    wins = {}
    for nm, w in rr["controls"]["windows"].items():
        if w.get("excluded_by_config"):
            wins[nm] = {"valid": False, "excluded_by_config": True}
        else:
            wins[nm] = {"valid": bool(w.get("valid")), "buccal": {"median_thickness_mm": w.get("buccal_median_mm"),
                                                                 "mean_thickness_mm": w.get("buccal_mean_mm")},
                        "rigid_shift_mm": w.get("rigid_shift_mm"), "dilation_mm": w.get("dilation_mm")}
    res["controls"]["windows"] = wins
    res["controls"]["background_offset_mm"] = rr["controls"]["background_mm"]
    res["controls"]["registration_accuracy"] = rr["controls"]["registration_accuracy"]
    ae = rr["as_executed"]
    res["volume"].update({"PRIMARY_net_bone_gain_mm3": ae["net_mm3"], "gain_before_implant_subtraction_mm3": ae["gross_mm3"],
                          "gain_before_background_and_implant_mm3": ae["gain_before_background_mm3"],
                          "implant_in_envelope_mm3": ae["implant_in_envelope_mm3"], "loss_mm3": ae["loss_mm3"],
                          "background_subtracted_mm": ae["background_mm"], "projected_area_mm2": ae["area_mm2"],
                          "coverage": ae["coverage"], "mean_thickness_mm": ae["mean_thickness_mm"]})
    return res


def main():
    case, post, ctrl_rule = sys.argv[1], sys.argv[2], sys.argv[3]
    assert ctrl_rule in ("old", "height", "heightfilter")
    depth_json = sys.argv[sys.argv.index("--depth-json") + 1] if "--depth-json" in sys.argv else \
        str(ROOT / "FINAL_ARCH_20260907/containment_NULLCORRECTED.json")
    tag = f"{post}_{ctrl_rule}" + (sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "")
    outj = SFX / case / f"measure_{tag}.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists; not overwritten")
    t0 = time.time()
    script_sha_at_start = sha256_file(Path(__file__).resolve())
    c = setup(case, post, depth_json)
    c.ext_depth = c.depth
    npz = {}
    out = {"case": case, "post": post, "post_file": str(post_path(case, post).relative_to(ROOT)),
           "post_sha256": sha256_file(post_path(case, post)), "ctrl_rule": ctrl_rule, "rules_sha256": RULES_SHA,
           "contained_depth_mm": c.depth, "depth_json": c.depth_json, "titanium_match": c.titanium_match,
           "implant_voxels": None if c.implant_idx is None else int(len(c.implant_idx)),
           "eligible_pins": c.elig, "readers": {}}
    for R in READERS:
        rr = run_reader(c, R, ctrl_rule, npz)
        if "--thr" in sys.argv and R == "A":
            rr["threshold_sweep_gross_mm3"] = threshold_sweep(c, R, rr["controls"]["background_unrounded"])
            rr["threshold_sweep_gross_mm3"]["450"] = rr["as_executed"]["gross_mm3"]
        out["readers"][R] = rr
        write_new(SFX / case / f"results_{tag}_{R}.json", results_like(c, R, rr, ctrl_rule))
        print(f"[{case}/{R} {tag}] bg {rr['controls']['background_mm']}  as-exec {rr['as_executed']['net_mm3']}  "
              f"primary {rr['primary']['net_mm3']}  margins {rr['primary']['margins']}  "
              f"ext {rr['extended']['whole_net_mm3']}  aborts {rr['controls']['abort_conditions']}", flush=True)
    out["peak_rss_GiB"] = round(rss_gib(), 2)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    pz = SFX / case / f"measure_{tag}_fields.npz"
    with open(pz, "xb") as fh:
        np.savez_compressed(fh, **npz)
    out["fields_npz"] = str(pz.relative_to(ROOT)); out["fields_npz_sha256"] = sha256_file(pz)
    out["script_sha256"] = script_sha_at_start
    out["script_sha256_at_end"] = sha256_file(Path(__file__).resolve())
    write_new(outj, out)
    print(f"[{case}] written {outj.relative_to(ROOT)}  peak {out['peak_rss_GiB']} GiB  {out['runtime_s']} s")


if __name__ == "__main__":
    main()

