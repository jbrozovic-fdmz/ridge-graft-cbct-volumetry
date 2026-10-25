#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "analysis"))
import roi_extent_sweep as SWEEP
import secondary_contained_depth as SCD
from pipeline import geom, landmarks, volume as V

TAG = "2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
GRID_STEP = 0.20
WIDE_Y0 = 11.0
RULE_PY = ROOT / "analysis/pin_artefact_rule_2026-09-14.py"
RULE_JSON = ROOT / "FINAL_ARCH_20260907/pin_artefact_rule_2026-09-14.json"
DECL_JSON = ROOT / "FINAL_ARCH_20260907/pin_artefact_rule_DECLARATION_2026-09-14.json"
FROZEN = ROOT / "FINAL_ARCH_20260907/FROZEN_RESULTS.json"
OUT_FINAL = ROOT / f"FINAL_ARCH_20260907/pin_artefact_corrected_{TAG}.json"
RULE_TEXT_SHA_PREFIX = "e3b0c442"
EXPECTED_NET = {"S1": {"A": 105.6, "B": 104.8}, "S2": {"A": 119.9, "B": 108.0},
                "S3": {"A": 139.5, "B": 140.3}, "S4": {"A": 316.1, "B": 332.2}}
EXPECTED_SECONDARY = {"S4": {"A": 360.6, "B": 379.8}}
DECOMP_ONLY_D = 0.0
QC_PINS = {"S2": (64, 63)}


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def r(x, n=3):
    return None if x is None else round(float(x), n)


def write_new(p, obj):
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def load_rule():
    rule = json.loads(RULE_JSON.read_text())
    decl = json.loads(DECL_JSON.read_text())
    spec = importlib.util.spec_from_file_location("pin_artefact_rule_20260914", RULE_PY)
    RULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RULE)
    h = sha256_text(rule["rule_text"])
    assert h == rule["rule_text_sha256"] == decl["rule_text_sha256"] == sha256_text(decl["rule_text"]) \
        == sha256_text(RULE.RULE_TEXT), "ABORT: rule text changed since declaration"
    assert h.startswith(RULE_TEXT_SHA_PREFIX), h
    s = sha256_file(RULE_PY)
    assert s == rule["script_sha256"] == decl["script_sha256_at_declaration"], \
        "ABORT: rule script changed since declaration"
    d = float(rule["d_mm"])
    sens = [float(v) for v in rule["sensitivity_d_mm"]]
    assert sens == sorted({round(max(0.0, d - 0.3), 1), d, round(d + 0.3, 1)}), sens
    assert rule["window_volumes_computed"] is False and decl["window_volumes_computed"] is False
    frz = json.loads(FROZEN.read_text())["cases"]
    for c in CASES:
        for R in READERS:
            assert frz[c][R]["net_mm3"] == EXPECTED_NET[c][R] == \
                rule["frozen_values_to_reproduce_before_correction"][c][R]["net_mm3"]
            assert frz[c][R]["gross_mm3"] == rule["frozen_values_to_reproduce_before_correction"][c][R]["gross_mm3"]
    info = {"rule_json": str(RULE_JSON.relative_to(ROOT)), "rule_json_file_sha256": sha256_file(RULE_JSON),
            "rule_text_sha256": h, "rule_text_sha256_verified_json_decl_module": True,
            "rule_script_sha256": s, "rule_script_sha256_verified": True,
            "declared_utc": decl["declared_utc"], "d_mm": d, "sensitivity_d_mm": sens}
    return RULE, rule, info


def envelope_overlap(vox_SAL, frame, Hpre, Hpost, xe, ye, vx):
    if vox_SAL is None or not len(vox_SAL):
        return 0.0
    P = frame.to_local(np.asarray(vox_SAL, float) * vx)
    ny, nx = Hpre.shape
    ix = np.digitize(P[:, 0], xe) - 1
    iy = np.digitize(P[:, 1], ye) - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not inside.any():
        return 0.0
    lo = Hpre[iy[inside], ix[inside]]
    hi = Hpost[iy[inside], ix[inside]]
    z = P[inside, 2]
    half = vx / 2.0
    frac = np.clip(np.minimum(hi, z + half) - np.maximum(lo, z - half), 0.0, vx) / vx
    frac = np.where(np.isfinite(lo) & np.isfinite(hi), frac, 0.0)
    return float(frac.sum()) * vx ** 3


def term(vox, frame, Hpre, Hpost, xe, ye, vx):
    if vox is None or not len(vox):
        return 0.0, 0.0
    t = V.implant_volume_in_envelope(vox, frame, Hpre, Hpost, xe, ye, vx)["implant_in_envelope_mm3"]
    u = envelope_overlap(vox, frame, Hpre, Hpost, xe, ye, vx)
    assert t == round(u, 1), (t, u)
    return t, u


def gain_unrounded(Hpre, Hpost, background, step=GRID_STEP):
    t = Hpost - Hpre - background
    ok = np.isfinite(t)
    t = t[ok]
    return float(t[t > 0].sum() * step * step)


class Ctx:
    pass


def setup(case, RULE):
    c = Ctx()
    c.case = case
    au = json.loads(Path(f"analysis/{case}/fixation_pins_2026-09-13.json").read_text())
    c.lo = np.array(au["crop_lo_vox"]); c.hi = np.array(au["crop_hi_vox"])
    c.sl = tuple(slice(int(a), int(b)) for a, b in zip(c.lo, c.hi))
    c.res = {"A": json.loads(Path(f"analysis/{case}/results.json").read_text()),
             "B": json.loads(Path(f"analysis/{case}/readerB/results.json").read_text())}
    c.frozen = json.loads(FROZEN.read_text())["cases"][case]
    c.excluded_cfg = set(json.loads(Path(f"cases/{case}.json").read_text()).get("exclude_control_windows") or [])
    c.frames = {R: geom.LocalFrame.from_dict(c.res[R]["frame"]) for R in READERS}
    c.bands = {R: [float(v) for v in c.res[R]["search_band_Z_mm"]] for R in READERS}
    c.depth_sec = SCD.contained_depth(case)
    avail = RULE.wait_for_memory()
    c.mem_available_GiB_at_load = avail
    c.post, c.vx = RULE.read_crop(ROOT / f"cbct_blinded/{case}/post_reg.nii.gz", c.sl)
    pre, vx2 = RULE.read_crop(ROOT / f"analysis/{case}/pre_cal.nii.gz", c.sl)
    assert abs(c.vx - vx2) < 1e-6 and abs(c.vx - au["voxel_mm"]) < 1e-6
    c.origin_mm = c.lo * c.vx
    print(f"[{case}] crop {c.post.shape} vx {c.vx:.3f}  RSS {rss_gib():.2f} GiB", flush=True)
    c.lbl, c.pins, c.implant_label, c.pin_rows = RULE.pin_components(case, c.post, pre, c.lo, c.vx)
    imp = c.res["A"]["implant"]
    c.implant_idx = None
    if imp:
        for R in READERS:
            assert c.res[R]["implant"]["bbox_min_SAL_vox"] == imp["bbox_min_SAL_vox"]
            assert c.res[R]["implant"]["n_voxels"] == imp["n_voxels"]
        c.implant_idx = np.argwhere(c.lbl == c.implant_label) + c.lo
        assert len(c.implant_idx) == imp["n_voxels"]
    m0 = np.isin(c.lbl, c.pins) if c.pins else np.zeros(c.lbl.shape, bool)
    c.pin_idx = np.argwhere(m0) + c.lo
    del m0
    pf = geom.bone_field(pre)
    Vp, Fp = geom.iso_surface(pf, c.vx)
    del pf, pre
    Vp = Vp + c.origin_mm
    c.KEEP, c.P = {}, {}
    for R in READERS:
        z0, z1 = c.bands[R]
        y_hi = c.frames[R].offset[1] + max(SWEEP.DEPTHS) + SWEEP.PAD_Y
        c.KEEP[R] = geom.window_box({"X_mm": [-40.0, 60.0], "Y_mm": [WIDE_Y0, y_hi]}, {}, z0, z1,
                                    pad_x_mm=0.0, pad_y_mm=0.0)
        c.P[R] = geom.surface_points_local(Vp, Fp, c.frames[R], c.KEEP[R])
    del Vp, Fp
    return c


def post_clouds(c, vol):
    f = geom.bone_field(vol)
    Vq, Fq = geom.iso_surface(f, c.vx)
    del f
    Vq = Vq + c.origin_mm
    Q = {R: geom.surface_points_local(Vq, Fq, c.frames[R], c.KEEP[R]) for R in READERS}
    del Vq, Fq
    return Q


def measure_reader(c, R, Q, pin_idx):
    res = c.res[R]; frame = c.frames[R]; z0, z1 = c.bands[R]; roi = res["roi"]; P = c.P[R]
    cw = res["control_windows"]
    valid = [nm for nm, d in res["controls"]["windows"].items()
             if d.get("valid", True) and not d.get("excluded_by_config") and nm not in c.excluded_cfg]
    ctrl, bg_vals = {}, []
    for nm in valid:
        b = V.measure_window(P, Q, cw[nm], roi["Y_mm"], GRID_STEP, z0, z1, keep_box=c.KEEP[R])
        assert b.get("cells", 0) >= 200
        ctrl[nm] = r(b["median_thickness_mm"], 4)
        bg_vals.append(b["median_thickness_mm"])
    background = float(np.mean(bg_vals))
    main = V.measure_window(P, Q, roi["X_mm"], roi["Y_mm"], GRID_STEP, z0, z1,
                            background=background, keep_box=c.KEEP[R])
    Hpre, Hpost, xe, ye = main["_Hpre"], main["_Hpost"], main["_x_edges"], main["_y_edges"]
    imp_t, imp_u = term(c.implant_idx, frame, Hpre, Hpost, xe, ye, c.vx)
    pin_t, pin_u = term(pin_idx, frame, Hpre, Hpost, xe, ye, c.vx) if pin_idx is not None else (0.0, 0.0)
    out = {"background_mm": r(background, 3), "background_unrounded": float(background),
           "control_median_thickness_mm": ctrl, "controls_used": valid,
           "gross_mm3": main["gain_mm3"], "implant_in_envelope_mm3": imp_t,
           "pin_in_envelope_mm3": pin_t,
           "net_mm3": round(main["gain_mm3"] - imp_t - pin_t, 1),
           "net_without_R4_mm3": round(main["gain_mm3"] - imp_t, 1),
           "coverage": main["coverage"], "area_mm2": main["area_mm2"],
           "mean_thickness_mm": main["mean_thickness_mm"],
           "_implant_unr": imp_u, "_pin_unr": pin_u,
           "_Hpre": Hpre, "_Hpost": Hpost, "_xe": xe, "_ye": ye}
    if c.depth_sec != 10.0:
        roi_w = landmarks.standard_roi(10.0, 10.0, offset=frame.offset, anchor="preop_anatomy")
        ctrl_s = landmarks.control_windows_clean(roi_w, res["new_bone_md_profile"], gap_mm=SWEEP.GAP,
                                                 frac=SWEEP.FRAC, md_positive_is=frame.md_positive_is)
        ctrl_s.pop("_placement"); ctrl_s.pop("_rule")
        roi_s = landmarks.standard_roi(10.0, c.depth_sec, offset=frame.offset, anchor="preop_anatomy")
        sbg, used = [], []
        for nm, xr in ctrl_s.items():
            if nm in c.excluded_cfg:
                continue
            b = V.measure_window(P, Q, xr, roi_s["Y_mm"], SWEEP.STEP, z0, z1, keep_box=c.KEEP[R])
            if b.get("cells", 0) < 200 or abs(b["median_thickness_mm"]) > SWEEP.TOL:
                continue
            sbg.append(b["median_thickness_mm"]); used.append(nm)
        sbackground = float(np.mean(sbg)) if sbg else 0.0
        ms = V.measure_window(P, Q, roi_s["X_mm"], roi_s["Y_mm"], SWEEP.STEP, z0, z1,
                              background=sbackground, keep_box=c.KEEP[R])
        sub_full = res["volume"]["implant_in_envelope_mm3"]
        sp_t, sp_u = (term(pin_idx, frame, ms["_Hpre"], ms["_Hpost"], ms["_x_edges"], ms["_y_edges"], c.vx)
                      if pin_idx is not None else (0.0, 0.0))
        simp = sub_full * c.depth_sec / 10.0
        out["secondary"] = {"depth_mm": c.depth_sec, "background_mm": r(sbackground, 3),
                            "controls_used": used, "gross_mm3": ms["gain_mm3"],
                            "implant_term_scaled_mm3": r(simp, 3), "pin_in_envelope_mm3": sp_t,
                            "net_mm3": round(ms["gain_mm3"] - simp - sp_t, 1),
                            "net_without_R4_mm3": round(ms["gain_mm3"] - simp, 1),
                            "coverage": ms["coverage"],
                            "_bg": sbackground, "_pin_unr": sp_u,
                            "_Hpre": ms["_Hpre"], "_Hpost": ms["_Hpost"]}
    return out


def assert_reproduces(c, R, m):
    res = c.res[R]; vol = res["volume"]; frz = c.frozen[R]
    assert m["net_mm3"] == frz["net_mm3"] == vol["PRIMARY_net_bone_gain_mm3"] == EXPECTED_NET[c.case][R], \
        (c.case, R, m["net_mm3"], frz["net_mm3"])
    assert m["gross_mm3"] == frz["gross_mm3"] == vol["gain_before_implant_subtraction_mm3"]
    assert m["implant_in_envelope_mm3"] == vol["implant_in_envelope_mm3"]
    assert m["background_mm"] == res["controls"]["background_offset_mm"]
    assert m["area_mm2"] == vol["projected_area_mm2"] and m["coverage"] == vol["coverage"]
    assert m["mean_thickness_mm"] == vol["mean_thickness_mm"]
    if "secondary" in m:
        stored = json.loads(Path(f"analysis/{c.case}/secondary_contained_depth_{R}.json").read_text())
        assert m["secondary"]["net_mm3"] == stored["secondary_10xD_mm3"] == EXPECTED_SECONDARY[c.case][R], \
            (m["secondary"]["net_mm3"], stored["secondary_10xD_mm3"])
    else:
        assert c.case not in EXPECTED_SECONDARY
    print(f"  {c.case}/{R}: unmodified net {m['net_mm3']} == frozen {frz['net_mm3']}"
          + (f"; 10x{c.depth_sec:g} {m['secondary']['net_mm3']} == stored" if "secondary" in m else "")
          + "  OK", flush=True)


def public(m):
    o = {k: v for k, v in m.items() if not k.startswith("_")}
    if "secondary" in m:
        o["secondary"] = {k: v for k, v in m["secondary"].items() if not k.startswith("_")}
    return o


def gate(case):
    RULE, rule, info = load_rule()
    p = ROOT / f"analysis/{case}/pin_artefact_gate_{TAG}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; not overwritten")
    t0 = time.time()
    c = setup(case, RULE)
    Q = post_clouds(c, c.post)
    out = {"case": case, "_what": "",
           "rule": info, "pins": c.pin_rows, "readers": {}}
    for R in READERS:
        m = measure_reader(c, R, Q[R], None)
        assert_reproduces(c, R, m)
        out["readers"][R] = public(m)
    out["all_assertions_passed"] = True
    out["peak_rss_GiB"] = r(rss_gib(), 2)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_new(p, out)
    print(f"[{case}] gate written; peak RSS {out['peak_rss_GiB']} GiB", flush=True)


def cells_changed(H0, H1):
    same = (np.isnan(H0) & np.isnan(H1)) | (H0 == H1)
    return ~same


def decompose(c, R, base, corr):
    H0pre, H0post, H1pre, H1post = base["_Hpre"], base["_Hpost"], corr["_Hpre"], corr["_Hpost"]
    assert np.array_equal(H0pre, H1pre, equal_nan=True), "pre field changed (pre_cal is never touched)"
    bg0, bg1 = base["background_unrounded"], corr["background_unrounded"]
    g00 = gain_unrounded(H0pre, H0post, bg0)
    g10 = gain_unrounded(H0pre, H1post, bg0)
    g11 = gain_unrounded(H0pre, H1post, bg1)
    t0 = H0post - H0pre - bg0; t1 = H1post - H0pre - bg0
    c0 = np.where(np.isfinite(t0), np.clip(t0, 0, None), 0.0); c1 = np.where(np.isfinite(t1), np.clip(t1, 0, None), 0.0)
    dc = (c1 - c0) * GRID_STEP ** 2
    ch = cells_changed(H0post, H1post)
    frame = c.frames[R]; xe, ye = base["_xe"], base["_ye"]
    pin_on_frozen = envelope_overlap(c.pin_idx, frame, H0pre, H0post, xe, ye, c.vx)
    trace = g10 - g00; bgt = g11 - g10
    imp_change = corr["_implant_unr"] - base["_implant_unr"]
    metal = corr["_pin_unr"]
    total_unr = trace + bgt - imp_change - metal
    return {"delta_net_reported_mm3": round(corr["net_mm3"] - base["net_mm3"], 1),
            "delta_net_unrounded_mm3": r(total_unr, 3),
            "artefact_surface_total_mm3": r(trace + bgt, 3),
            "trace_term_background_held_mm3": r(trace, 3),
            "trace_term_cells_lowered_mm3": r(dc[dc < 0].sum(), 3),
            "trace_term_cells_raised_mm3": r(dc[dc > 0].sum(), 3),
            "background_term_mm3": r(bgt, 3),
            "metal_volume_term_mm3": r(-metal, 3),
            "implant_term_change_mm3": r(-imp_change, 3),
            "pin_metal_in_envelope_on_FROZEN_fields_mm3_(2026-09-13_figure)": r(pin_on_frozen, 3),
            "cells_Hpost_changed": int(ch.sum()),
            "cells_changed_NaN_status": int((np.isnan(H0post) != np.isnan(H1post)).sum()),
            "cells_measured_before_after": [int(np.isfinite(H0post - H0pre).sum()),
                                            int(np.isfinite(H1post - H1pre).sum())],
            "mean_Hpost_change_in_changed_cells_mm": (r(np.nanmean((H1post - H0post)[ch]), 3)
                                                      if np.isfinite((H1post - H0post)[ch]).any() else None),
            "min_max_Hpost_change_mm": ([r(np.nanmin(H1post - H0post), 3), r(np.nanmax(H1post - H0post), 3)]
                                        if np.isfinite(H1post - H0post).any() else None)}


def decompose_secondary(base, corr):
    b, k = base["secondary"], corr["secondary"]
    assert np.array_equal(b["_Hpre"], k["_Hpre"], equal_nan=True)
    g00 = gain_unrounded(b["_Hpre"], b["_Hpost"], b["_bg"])
    g10 = gain_unrounded(b["_Hpre"], k["_Hpost"], b["_bg"])
    g11 = gain_unrounded(b["_Hpre"], k["_Hpost"], k["_bg"])
    return {"delta_net_reported_mm3": round(k["net_mm3"] - b["net_mm3"], 1),
            "trace_term_background_held_mm3": r(g10 - g00, 3), "background_term_mm3": r(g11 - g10, 3),
            "metal_volume_term_mm3": r(-k["_pin_unr"], 3),
            "cells_Hpost_changed": int(cells_changed(b["_Hpost"], k["_Hpost"]).sum())}


def column_qc(c, fields):
    """reader-A Hpost in the column of
    each QC pin's centroid X. over = cells with |Y - pin centroid Y| <= 1.4 mm; neighbours =
    1.4 < |Y - Y_c| <= 2.6 mm. protrusion = max(over) - median(neighbours); beyond_titanium =
    max(over) - max Z of that pin's titanium voxels within 0.3 mm of the column centre in X."""
    out = {}
    frame = c.frames["A"]
    for label in QC_PINS.get(c.case, ()):
        idx = np.argwhere(c.lbl == label) + c.lo
        L = frame.to_local(idx.astype(float) * c.vx)
        cen = L.mean(axis=0)
        k0 = next(iter(fields))
        xe, ye = fields[k0]["_xe"], fields[k0]["_ye"]
        xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
        ix = int(np.digitize(cen[0], xe) - 1)
        near = np.abs(L[:, 0] - xc[ix]) <= 0.3
        ti_z = float(L[near, 2].max()) if near.any() else float(L[:, 2].max())
        rows = np.nonzero(np.abs(yc - cen[1]) <= 2.6)[0]
        over = np.abs(yc[rows] - cen[1]) <= 1.4
        o = {"column_X_mm": r(xc[ix], 2), "pin_centroid_Y_mm": r(cen[1], 2),
             "titanium_max_Z_in_column_mm": r(ti_z, 2), "Y_mm": [r(yc[j], 2) for j in rows],
             "Hpre": [r(fields[k0]["_Hpre"][j, ix], 2) for j in rows], "variants": {}}
        for k, f in fields.items():
            H = f["_Hpost"][rows, ix]
            mo = np.nanmax(H[over]) if np.isfinite(H[over]).any() else np.nan
            mn = np.nanmedian(H[~over]) if np.isfinite(H[~over]).any() else np.nan
            o["variants"][k] = {"Hpost": [r(v, 2) for v in H], "max_over_pin_mm": r(mo, 2),
                                "median_neighbours_mm": r(mn, 2), "protrusion_mm": r(mo - mn, 2),
                                "max_over_pin_minus_titanium_Z_mm": r(mo - ti_z, 2)}
        out[f"pin{label}"] = o
    return out


def apply(case):
    RULE, rule, info = load_rule()
    gates = {}
    for cc in CASES:
        gp = ROOT / f"analysis/{cc}/pin_artefact_gate_{TAG}.json"
        assert gp.exists(), f"ABORT: gate file missing for {cc}; run --gate for all four cases first"
        g = json.loads(gp.read_text())
        assert g["all_assertions_passed"] is True and g["rule"]["rule_text_sha256"] == info["rule_text_sha256"]
        for R in READERS:
            assert g["readers"][R]["net_mm3"] == EXPECTED_NET[cc][R]
        gates[cc] = {R: g["readers"][R]["net_mm3"] for R in READERS}
    p = ROOT / f"analysis/{case}/pin_artefact_corrected_{TAG}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; not overwritten")
    t0 = time.time()
    c = setup(case, RULE)
    Q = post_clouds(c, c.post)
    base = {}
    for R in READERS:
        base[R] = measure_reader(c, R, Q[R], None)
        assert_reproduces(c, R, base[R])
    del Q
    out = {"case": case, "_what": "",
           "rule": info, "gate_all_eight_frozen_nets_reproduced": gates, "pins": c.pin_rows,
           "implant_label": c.implant_label, "n_pin_titanium_voxels": int(len(c.pin_idx)),
           "mem_available_GiB_at_load": c.mem_available_GiB_at_load,
           "reproduced_unmodified": {R: public(base[R]) for R in READERS}, "variants": {}}
    d_list = [DECOMP_ONLY_D] + info["sensitivity_d_mm"]
    qc_fields = {"frozen": base["A"]}
    for d in d_list:
        key = f"d={d:.1f}"
        v = {"d_mm": d, "role": ("PRIMARY" if d == info["d_mm"] else
                                 "decomposition component (titanium only, no margin), NOT an estimate"
                                 if d == DECOMP_ONLY_D else "sensitivity")}
        if not c.pins:
            v["note"] = ""
            corr = {R: base[R] for R in READERS}
            v["inpainting"] = {"mask_voxels": 0}
        else:
            M = RULE.pin_mask(c.lbl, c.pins, c.vx, d)
            filled = RULE.inpaint(c.post, M)
            faces = any(M.take(0, axis=a).any() or M.take(-1, axis=a).any() for a in range(3))
            v["inpainting"] = {
                "mask_voxels": int(M.sum()), "mask_mm3": r(M.sum() * c.vx ** 3, 3),
                "mask_touches_crop_face": bool(faces),
                "mask_voxels_grey_gt_450_before": int((c.post[M] > geom.ISO_LEVEL).sum()),
                "mask_voxels_grey_gt_450_after": int((filled[M] > geom.ISO_LEVEL).sum()),
                "filled_voxels_copied_from_metal_gt_2800": int((filled[M] > geom.METAL_LEVEL).sum()),
                "voxels_changed_outside_mask": int((filled[~M] != c.post[~M]).sum()),
                "filled_crop_sha256": hashlib.sha256(np.ascontiguousarray(filled).tobytes()).hexdigest()}
            assert v["inpainting"]["voxels_changed_outside_mask"] == 0
            del M
            Qc = post_clouds(c, filled)
            del filled
            corr = {R: measure_reader(c, R, Qc[R], c.pin_idx) for R in READERS}
            del Qc
        v["readers"] = {}
        for R in READERS:
            m = corr[R]
            rec = public(m)
            rec["delta_net_vs_frozen_mm3"] = round(m["net_mm3"] - base[R]["net_mm3"], 1)
            rec["delta_net_without_R4_vs_frozen_mm3"] = round(m["net_without_R4_mm3"] - base[R]["net_mm3"], 1)
            rec["coverage_frozen"] = base[R]["coverage"]
            rec["background_frozen_mm"] = base[R]["background_mm"]
            rec["decomposition"] = decompose(c, R, base[R], m) if c.pins else None
            if "secondary" in m:
                rec["secondary"]["frozen_net_mm3"] = base[R]["secondary"]["net_mm3"]
                rec["secondary"]["delta_net_vs_frozen_mm3"] = round(m["secondary"]["net_mm3"]
                                                                    - base[R]["secondary"]["net_mm3"], 1)
                rec["secondary"]["decomposition"] = decompose_secondary(base[R], m) if c.pins else None
            v["readers"][R] = rec
            print(f"  {case}/{R} {key}: net {m['net_mm3']} (w/o R4 {m['net_without_R4_mm3']}), gross "
                  f"{m['gross_mm3']}, pin {m['pin_in_envelope_mm3']}, bg {m['background_mm']}, cov {m['coverage']}"
                  + (f" | 10x{c.depth_sec:g} {m['secondary']['net_mm3']}" if "secondary" in m else ""),
                  flush=True)
        if c.pins:
            qc_fields[key] = corr["A"]
        out["variants"][key] = v
        print(f"  {key} done  RSS {rss_gib():.2f} GiB", flush=True)
    if case in QC_PINS:
        out["R7_column_qc_readerA"] = column_qc(c, qc_fields)
        out["R7_column_qc_readerA"]["_definition"] = column_qc.__doc__.split("\n", 1)[0] + " " + \
            " ".join(column_qc.__doc__.split("\n", 1)[1].split())
    out["peak_rss_GiB"] = r(rss_gib(), 2)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_new(p, out)
    print(f"[{case}] written {p.relative_to(ROOT)}  ({out['runtime_s']} s, peak RSS {out['peak_rss_GiB']} GiB)",
          flush=True)


def collect():
    RULE, rule, info = load_rule()
    per = {}
    for c in CASES:
        g = json.loads((ROOT / f"analysis/{c}/pin_artefact_gate_{TAG}.json").read_text())
        a = json.loads((ROOT / f"analysis/{c}/pin_artefact_corrected_{TAG}.json").read_text())
        assert g["all_assertions_passed"] and a["rule"]["rule_text_sha256"] == info["rule_text_sha256"]
        per[c] = a
    d = info["d_mm"]
    table = []
    for c in CASES:
        for R in READERS:
            a = per[c]
            row = {"case": c, "reader": R, "frozen_net_mm3": a["reproduced_unmodified"][R]["net_mm3"]}
            for k, v in a["variants"].items():
                rr = v["readers"][R]
                tag = k.replace("d=", "d")
                row[f"net_{tag}_mm3"] = rr["net_mm3"]
                if v["d_mm"] == d:
                    row.update({"PRIMARY_corrected_net_mm3": rr["net_mm3"],
                                "delta_mm3": rr["delta_net_vs_frozen_mm3"],
                                "delta_pct": round(100 * rr["delta_net_vs_frozen_mm3"] / row["frozen_net_mm3"], 1),
                                "net_without_R4_mm3": rr["net_without_R4_mm3"],
                                "pin_in_envelope_mm3": rr["pin_in_envelope_mm3"],
                                "gross_mm3": rr["gross_mm3"],
                                "implant_in_envelope_mm3": rr["implant_in_envelope_mm3"],
                                "coverage_frozen_to_corrected": [rr["coverage_frozen"], rr["coverage"]],
                                "background_frozen_to_corrected_mm": [rr["background_frozen_mm"], rr["background_mm"]],
                                "cells_Hpost_changed": (rr["decomposition"] or {}).get("cells_Hpost_changed", 0),
                                "decomposition": rr["decomposition"]})
                    if "secondary" in rr:
                        row["secondary_10x14"] = {"frozen_mm3": rr["secondary"]["frozen_net_mm3"],
                                                  **{f"net_{kk.replace('d=', 'd')}_mm3":
                                                     vv["readers"][R]["secondary"]["net_mm3"]
                                                     for kk, vv in a["variants"].items()},
                                                  "decomposition_primary_d": rr["secondary"]["decomposition"]}
            table.append(row)
    fin = {"_what": "",
           "date": TAG, "rule": info,
           "gate": "",
           "pairs_2x2": {"implant": ["", ""], "no_implant": ["", ""]},
           "table": table,
           "R7_S2_column_qc_readerA": per["S2"].get("R7_column_qc_readerA"),
           "inpainting_qc": {c: {k: v["inpainting"] for k, v in per[c]["variants"].items()} for c in CASES},
           "peak_rss_GiB_apply": {c: per[c]["peak_rss_GiB"] for c in CASES},
           "per_case_files": [f"analysis/{c}/pin_artefact_corrected_{TAG}.json" for c in CASES],
           "script": "analysis/pin_artefact_apply_2026-09-14.py",
           "script_sha256": sha256_file(Path(__file__).resolve())}
    write_new(OUT_FINAL, fin)
    for row in table:
        print({k: v for k, v in row.items() if k not in ("decomposition",)})


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--gate" and a[1] in CASES:
        gate(a[1])
    elif len(a) == 2 and a[0] == "--apply" and a[1] in CASES:
        apply(a[1])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit(__doc__)

