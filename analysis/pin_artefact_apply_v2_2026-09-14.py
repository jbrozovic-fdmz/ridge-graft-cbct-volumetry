#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "analysis"))
from pipeline import landmarks, volume as V

TAG = "v2_2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
A1_PY = ROOT / "analysis/pin_artefact_apply_2026-09-14.py"
A1_SHA = "552187e484a8d11a59cf9cb7190c930bd0c6488da30c8c6eb33748a8389d0875"
RULE2_PY = ROOT / "analysis/pin_artefact_rule_v2_2026-09-14.py"
RULE2_JSON = ROOT / "FINAL_ARCH_20260907/pin_artefact_rule_v2_2026-09-14.json"
RULE2_TEXT_PREFIX = "5273a471"
RULE2_SCRIPT_PREFIX = "b7352626"
OUT_FINAL = ROOT / f"FINAL_ARCH_20260907/pin_artefact_corrected_{TAG}.json"
QC_SPUR_RADIUS_MM = 1.0


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def write_new(p, obj):
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_all():
    assert sha256_file(A1_PY) == A1_SHA, "ABORT: attempt-1 apply script (measurement path) changed"
    A1 = _load("pin_artefact_apply_20260914", A1_PY)
    RULE1, _, info1 = A1.load_rule()
    rule2 = json.loads(RULE2_JSON.read_text())
    RULE2 = _load("pin_artefact_rule_v2_20260914", RULE2_PY)
    h = sha256_text(rule2["rule_text"])
    assert h == rule2["rule_text_sha256"] == sha256_text(RULE2.RULE_TEXT), "ABORT: v2 rule text changed"
    assert h.startswith(RULE2_TEXT_PREFIX), h
    s = sha256_file(RULE2_PY)
    assert s == rule2["script_sha256_at_declaration"] and s.startswith(RULE2_SCRIPT_PREFIX), \
        "ABORT: v2 rule script changed since declaration"
    ps = sha256_text(json.dumps(RULE2.PARAMS, sort_keys=True))
    assert ps == rule2["parameters_sha256"] and RULE2.PARAMS == rule2["parameters"], "ABORT: parameters changed"
    assert rule2["volumes_loaded"] is False and rule2["corrected_values_computed"] is False
    elig, _ = RULE2.eligible_pins()
    assert elig == rule2["eligible_pins_per_case"] == RULE2.DECLARED_ELIGIBLE, elig
    d = float(rule2["parameters"]["d_mm"])
    dl = [float(x) for x in rule2["parameters"]["sensitivity_d_mm"]]
    assert d == 1.4 and dl == [1.0, 1.4, 2.0]
    assert RULE2.PARAMS["grid_step_mm"] == A1.GRID_STEP == 0.20
    for c in CASES:
        for R in READERS:
            assert rule2["frozen_values_to_reproduce_before_correction"][c][R]["net_mm3"] == A1.EXPECTED_NET[c][R]
    assert rule2["frozen_secondary_S4_10x14_to_reproduce"] == A1.EXPECTED_SECONDARY["S4"]
    info = {"rule_json": str(RULE2_JSON.relative_to(ROOT)), "rule_json_file_sha256": sha256_file(RULE2_JSON),
            "rule_text_sha256": h, "rule_text_sha256_verified_json_module": True,
            "rule_script_sha256": s, "rule_script_sha256_verified": True,
            "parameters_sha256": ps, "declared_utc": rule2["declared_utc"],
            "d_mm": d, "sensitivity_d_mm": dl, "eligible_pins_per_case": elig,
            "measurement_path_script": str(A1_PY.relative_to(ROOT)), "measurement_path_script_sha256": A1_SHA,
            "attempt1_rule_text_sha256_verified_for_helpers": info1["rule_text_sha256"]}
    return A1, RULE1, RULE2, rule2, info


def gate(case):
    A1, RULE1, RULE2, rule2, info = load_all()
    p = ROOT / f"analysis/{case}/pin_artefact_gate_{TAG}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; not overwritten")
    t0 = time.time()
    c = A1.setup(case, RULE1)
    Q = A1.post_clouds(c, c.post)
    out = {"case": case, "_what": "", "rule": info, "pins": c.pin_rows,
           "mem_available_GiB_at_load": c.mem_available_GiB_at_load, "readers": {}}
    for R in READERS:
        m = A1.measure_reader(c, R, Q[R], None)
        A1.assert_reproduces(c, R, m)
        out["readers"][R] = A1.public(m)
    out["all_assertions_passed"] = True
    out["peak_rss_GiB"] = r(A1.rss_gib(), 2)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_new(p, out)
    print(f"[{case}] v2 gate written; peak RSS {out['peak_rss_GiB']} GiB", flush=True)


def windows_for(c, R, base, A1):
    res = c.res[R]; frame = c.frames[R]; roi = res["roi"]; step = A1.GRID_STEP
    wins = {}
    xe, ye = base["_xe"], base["_ye"]
    assert np.array_equal(xe, V._edges(*roi["X_mm"], step)) and np.array_equal(ye, V._edges(*roi["Y_mm"], step))
    ctrl = {nm: (res["control_windows"][nm], roi["Y_mm"]) for nm in base["controls_used"]}
    wins["primary_10x10"] = {"Hpre": base["_Hpre"], "Hpost": base["_Hpost"], "xe": xe, "ye": ye,
                             "background": base["background_unrounded"], "implant_override": None,
                             "frozen_net": base["net_mm3"], "frozen_coverage": base["coverage"],
                             "controls": ctrl}
    if "secondary" in base:
        sb = base["secondary"]
        roi_s = landmarks.standard_roi(10.0, c.depth_sec, offset=frame.offset, anchor="preop_anatomy")
        xs, ys = V._edges(*roi_s["X_mm"], A1.SWEEP.STEP), V._edges(*roi_s["Y_mm"], A1.SWEEP.STEP)
        assert sb["_Hpre"].shape == (len(ys) - 1, len(xs) - 1)
        assert np.allclose(roi_s["Y_mm"], [roi["Y_mm"][0], roi["Y_mm"][0] + 14.0]) and np.allclose(roi_s["X_mm"], roi["X_mm"])
        roi_w = landmarks.standard_roi(10.0, 10.0, offset=frame.offset, anchor="preop_anatomy")
        cs = landmarks.control_windows_clean(roi_w, res["new_bone_md_profile"], gap_mm=A1.SWEEP.GAP,
                                             frac=A1.SWEEP.FRAC, md_positive_is=frame.md_positive_is)
        wins[f"secondary_10x{c.depth_sec:g}"] = {
            "Hpre": sb["_Hpre"], "Hpost": sb["_Hpost"], "xe": xs, "ye": ys, "background": sb["_bg"],
            "implant_override": res["volume"]["implant_in_envelope_mm3"] * c.depth_sec / 10.0,
            "frozen_net": sb["net_mm3"], "frozen_coverage": sb["coverage"],
            "controls": {nm: (cs[nm], roi_s["Y_mm"]) for nm in sb["controls_used"]}}
    return wins


def below_pre(Hpre, H, mask):
    d = (H - Hpre)[mask]
    d = d[np.isfinite(d)]
    return {"n_cells_below_Hpre": int((d < 0).sum()), "n_cells_more_than_0.3mm_below": int((d < -0.3).sum()),
            "min_H_minus_Hpre_mm": r(d.min(), 3) if len(d) else None}


def one_variant(A1, RULE2, c, R, w, pin_idx, pin_xy, per_pin, d):
    frame = c.frames[R]; vx = c.vx
    Hpre, Hpost, xe, ye, bg = w["Hpre"], w["Hpost"], w["xe"], w["ye"], w["background"]
    o = RULE2.corrected_window(V, frame, Hpre, Hpost, xe, ye, bg, pin_idx, c.implant_idx, vx, d,
                               implant_term_override=w["implant_override"])
    F, Hc = o["_F"], o["_Hpost_corrected"]
    assert np.array_equal(Hc[~F], Hpost[~F], equal_nan=True), "cells outside the footprint changed"
    dH = (Hc - Hpost)[F]
    dHf = dH[np.isfinite(dH)]
    g0 = A1.gain_unrounded(Hpre, Hpost, bg)
    g1 = A1.gain_unrounded(Hpre, Hc, bg)
    if w["implant_override"] is None:
        i0 = A1.envelope_overlap(c.implant_idx, frame, Hpre, Hpost, xe, ye, vx)
        i1 = A1.envelope_overlap(c.implant_idx, frame, Hpre, Hc, xe, ye, vx)
    else:
        i0 = i1 = float(w["implant_override"])
    pin_u = A1.envelope_overlap(pin_idx, frame, Hpre, Hc, xe, ye, vx) if pin_idx is not None else 0.0
    pin_frozen_u = A1.envelope_overlap(pin_idx, frame, Hpre, Hpost, xe, ye, vx) if pin_idx is not None else 0.0
    if pin_idx is not None:
        assert o["pin_term_mm3"] == round(pin_u, 1)
    t0 = Hpost - Hpre - bg; t1 = Hc - Hpre - bg
    c0 = np.where(np.isfinite(t0), np.clip(t0, 0, None), 0.0)
    c1 = np.where(np.isfinite(t1), np.clip(t1, 0, None), 0.0)
    dc = (c1 - c0) * A1.GRID_STEP ** 2
    surface_u = (g1 - g0) - (i1 - i0)
    rec = {"d_mm": d,
           "net_mm3": o["net_mm3"], "frozen_net_mm3": w["frozen_net"],
           "delta_net_mm3": round(o["net_mm3"] - w["frozen_net"], 1),
           "net_surface_fix_only_mm3": o["net_surface_fix_only_mm3"],
           "gross_mm3": o["gross_mm3"], "implant_term_mm3": o["implant_term_mm3"], "pin_term_mm3": o["pin_term_mm3"],
           "coverage_frozen_to_corrected": [w["frozen_coverage"], o["coverage"]], "area_mm2": o["area_mm2"],
           "decomposition_rounded": {
               "surface_term_mm3": round(o["net_surface_fix_only_mm3"] - w["frozen_net"], 1),
               "metal_term_mm3": -o["pin_term_mm3"]},
           "decomposition_unrounded": {
               "surface_term_mm3": r(surface_u, 3),
               "surface_term_cells_lowered_mm3": r(dc[dc < 0].sum(), 3),
               "surface_term_cells_raised_mm3": r(dc[dc > 0].sum(), 3),
               "implant_term_change_mm3": r(-(i1 - i0), 3),
               "metal_term_mm3": r(-pin_u, 3),
               "delta_net_mm3": r(surface_u - pin_u, 3),
               "background_term_mm3": 0.0,
               "eligible_pin_metal_on_FROZEN_fields_mm3_(reference,_not_additive)": r(pin_frozen_u, 3)},
           "footprint": dict(o["fill"]),
           "Hpost_change_in_footprint_mm": {
               "max_reduction_mm": r(-dHf.min(), 3) if len(dHf) else None,
               "max_raise_mm": r(dHf.max(), 3) if len(dHf) else None,
               "mean_change_mm": r(dHf.mean(), 3) if len(dHf) else None,
               "n_cells_lowered": int((dHf < 0).sum()), "n_cells_raised": int((dHf > 0).sum())},
           "corrected_below_Hpre_in_footprint": below_pre(Hpre, Hc, F),
           "frozen_below_Hpre_in_footprint": below_pre(Hpre, Hpost, F),
           "corrected_below_Hpre_whole_window": below_pre(Hpre, Hc, np.ones(F.shape, bool)),
           "footprint_cells_in_valid_control_windows_(reported_not_applied)": {
               nm: int(RULE2.footprint(V._edges(*xr, A1.GRID_STEP), V._edges(*yr, A1.GRID_STEP), pin_xy, d).sum())
               for nm, (xr, yr) in w["controls"].items()},
           "per_pin": {}}
    if c.implant_idx is not None:
        P = frame.to_local(np.asarray(c.implant_idx, float) * c.vx)
        ix = np.digitize(P[:, 0], xe) - 1; iy = np.digitize(P[:, 1], ye) - 1
        ins = (ix >= 0) & (ix < len(xe) - 1) & (iy >= 0) & (iy < len(ye) - 1)
        rec["implant_voxels_in_footprint_cells"] = int(F[iy[ins], ix[ins]].sum())
    for lb, (xy, zmax) in per_pin.items():
        Fp = RULE2.footprint(xe, ye, xy, d)
        near = RULE2.footprint(xe, ye, xy, QC_SPUR_RADIUS_MM)
        hf = Hpost[near]; hc = Hc[near]
        rec["per_pin"][str(lb)] = {
            "footprint_cells_this_pin": int(Fp.sum()),
            "cells_within_1.0mm": int(near.sum()),
            "titanium_max_Z_mm": r(zmax, 2),
            "frozen_max_Hpost_within_1.0mm_minus_titanium_max_Z_mm":
                r(np.nanmax(hf) - zmax, 2) if np.isfinite(hf).any() else None,
            "corrected_max_Hpost_within_1.0mm_minus_titanium_max_Z_mm":
                r(np.nanmax(hc) - zmax, 2) if np.isfinite(hc).any() else None}
    return rec, F, Hc


def apply(case):
    A1, RULE1, RULE2, rule2, info = load_all()
    gates = {}
    for cc in CASES:
        gp = ROOT / f"analysis/{cc}/pin_artefact_gate_{TAG}.json"
        assert gp.exists(), f"ABORT: v2 gate file missing for {cc}; run --gate for all four cases first"
        g = json.loads(gp.read_text())
        assert g["all_assertions_passed"] is True and g["rule"]["rule_text_sha256"] == info["rule_text_sha256"]
        for R in READERS:
            assert g["readers"][R]["net_mm3"] == A1.EXPECTED_NET[cc][R]
            if cc in A1.EXPECTED_SECONDARY:
                assert g["readers"][R]["secondary"]["net_mm3"] == A1.EXPECTED_SECONDARY[cc][R]
        gates[cc] = {R: g["readers"][R]["net_mm3"] for R in READERS}
    p = ROOT / f"analysis/{case}/pin_artefact_corrected_{TAG}.json"
    pz = ROOT / f"analysis/{case}/pin_artefact_fields_{TAG}.npz"
    for q in (p, pz):
        if q.exists():
            raise SystemExit(f"{q} exists; not overwritten")
    t0 = time.time()
    c = A1.setup(case, RULE1)
    Q = A1.post_clouds(c, c.post)
    del c.post
    base = {}
    for R in READERS:
        base[R] = A1.measure_reader(c, R, Q[R], None)
        A1.assert_reproduces(c, R, base[R])
    del Q
    elig = info["eligible_pins_per_case"][case]
    decl_rows = {row["label"]: row for row in rule2["pin_candidates_with_eligibility"][case]}
    got_rows = {row["label"]: row for row in c.pin_rows}
    assert set(got_rows) == set(decl_rows), (got_rows.keys(), decl_rows.keys())
    for lb in elig:
        assert decl_rows[lb]["eligible"] and got_rows[lb]["n_voxels"] == decl_rows[lb]["n_voxels"] \
            and got_rows[lb]["post_grey_max"] == round(decl_rows[lb]["post_grey_max"]), lb
    pin_vox = {lb: np.argwhere(c.lbl == lb) + c.lo for lb in elig}
    pin_idx = np.vstack([pin_vox[lb] for lb in elig]) if elig else None
    out = {"case": case, "_what": "",
           "rule": info, "gate_all_eight_frozen_nets_reproduced_v2": gates, "pins_all_candidates": c.pin_rows,
           "eligible_pins": elig, "eligible_pin_voxels": {str(lb): int(len(v)) for lb, v in pin_vox.items()},
           "implant_label": c.implant_label, "mem_available_GiB_at_load": c.mem_available_GiB_at_load,
           "reproduced_unmodified": {R: A1.public(base[R]) for R in READERS},
           "boundary_statement": "",
           "readers": {}}
    npz = {}
    for R in READERS:
        frame = c.frames[R]
        pin_xy = frame.to_local(pin_idx.astype(float) * c.vx)[:, :2] if pin_idx is not None else None
        per_pin = {}
        for lb in elig:
            L = frame.to_local(pin_vox[lb].astype(float) * c.vx)
            per_pin[lb] = (L[:, :2], float(L[:, 2].max()))
        wins = windows_for(c, R, base[R], A1)
        out["readers"][R] = {}
        for wn, w in wins.items():
            idt = RULE2.corrected_window(V, frame, w["Hpre"], w["Hpost"], w["xe"], w["ye"], w["background"],
                                         None, c.implant_idx, c.vx, info["d_mm"],
                                         implant_term_override=w["implant_override"])
            assert idt["net_mm3"] == w["frozen_net"] and np.array_equal(idt["_Hpost_corrected"], w["Hpost"],
                                                                         equal_nan=True), (wn, idt["net_mm3"])
            wrec = {"grid_shape": list(w["Hpre"].shape), "X_mm": [r(w["xe"][0], 2), r(w["xe"][-1], 2)],
                    "Y_mm": [r(w["ye"][0], 2), r(w["ye"][-1], 2)], "background_frozen_mm": r(w["background"], 4),
                    "frozen_net_mm3": w["frozen_net"], "identity_no_pin_reproduces_frozen": True, "variants": {}}
            key0 = f"{R}_{wn}"
            npz[f"{key0}_Hpre"] = w["Hpre"]; npz[f"{key0}_Hpost"] = w["Hpost"]
            npz[f"{key0}_xe"] = w["xe"]; npz[f"{key0}_ye"] = w["ye"]
            for d in info["sensitivity_d_mm"]:
                if pin_idx is None:
                    rec, F, Hc = one_variant(A1, RULE2, c, R, w, None, None, {}, d)
                    assert rec["net_mm3"] == w["frozen_net"] and np.array_equal(Hc, w["Hpost"], equal_nan=True)
                    rec["note"] = ""
                else:
                    rec, F, Hc = one_variant(A1, RULE2, c, R, w, pin_idx, pin_xy, per_pin, d)
                rec["role"] = "PRIMARY" if d == info["d_mm"] else "sensitivity"
                wrec["variants"][f"d={d:.1f}"] = rec
                npz[f"{key0}_d{d:.1f}_F"] = F; npz[f"{key0}_d{d:.1f}_Hc"] = Hc
                print(f"  {case}/{R} {wn} d={d:.1f}: net {rec['net_mm3']} (frozen {w['frozen_net']}; surface "
                      f"{rec['decomposition_unrounded']['surface_term_mm3']}, metal "
                      f"{rec['decomposition_unrounded']['metal_term_mm3']}); F {rec['footprint']['footprint_cells']} "
                      f"cells; max drop {rec['Hpost_change_in_footprint_mm']['max_reduction_mm']}; below Hpre "
                      f"{rec['corrected_below_Hpre_in_footprint']}", flush=True)
            out["readers"][R][wn] = wrec
    out["peak_rss_GiB"] = r(A1.rss_gib(), 2)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(pz, "xb") as fh:
        np.savez_compressed(fh, **npz)
    out["fields_npz"] = str(pz.relative_to(ROOT))
    out["fields_npz_sha256"] = sha256_file(pz)
    write_new(p, out)
    print(f"[{case}] written {p.relative_to(ROOT)} ({out['runtime_s']} s, peak RSS {out['peak_rss_GiB']} GiB)",
          flush=True)


def collect():
    A1, RULE1, RULE2, rule2, info = load_all()
    per = {}
    for c in CASES:
        g = json.loads((ROOT / f"analysis/{c}/pin_artefact_gate_{TAG}.json").read_text())
        a = json.loads((ROOT / f"analysis/{c}/pin_artefact_corrected_{TAG}.json").read_text())
        assert g["all_assertions_passed"] and a["rule"]["rule_text_sha256"] == info["rule_text_sha256"]
        assert sha256_file(ROOT / a["fields_npz"]) == a["fields_npz_sha256"]
        per[c] = a
    D = f"d={info['d_mm']:.1f}"
    table = []

    def newly_below(z, key0, dk):
        Hpre, Hpost = z[f"{key0}_Hpre"], z[f"{key0}_Hpost"]
        F, Hc = z[f"{key0}_{dk}_F"], z[f"{key0}_{dk}_Hc"]
        with np.errstate(invalid="ignore"):
            m = F & np.isfinite(Hc - Hpre) & (Hc < Hpre) & ~(Hpost < Hpre)
        dd = (Hc - Hpre)[m]
        return {"n_cells": int(m.sum()), "n_more_than_0.3mm_below": int((dd < -0.3).sum()),
                "min_H_minus_Hpre_mm": r(dd.min(), 3) if len(dd) else None}

    for c in CASES:
        a = per[c]
        z = np.load(ROOT / a["fields_npz"])
        for R in READERS:
            for wn, w in a["readers"][R].items():
                v = w["variants"]
                p = v[D]
                nb = {k: newly_below(z, f"{R}_{wn}", k.replace("=", "")) for k in v}
                table.append({
                    "case": c, "reader": R, "window": wn, "eligible_pins": a["eligible_pins"],
                    "frozen_net_mm3": w["frozen_net_mm3"],
                    "PRIMARY_corrected_net_mm3_d1.4": p["net_mm3"],
                    "delta_mm3": p["delta_net_mm3"],
                    "delta_pct": round(100 * p["delta_net_mm3"] / w["frozen_net_mm3"], 1),
                    "sensitivity_net_mm3": {k: vv["net_mm3"] for k, vv in v.items()},
                    "surface_term_mm3_unrounded": p["decomposition_unrounded"]["surface_term_mm3"],
                    "metal_term_mm3_unrounded": p["decomposition_unrounded"]["metal_term_mm3"],
                    "implant_term_change_mm3_unrounded": p["decomposition_unrounded"]["implant_term_change_mm3"],
                    "surface_fix_only_net_mm3": p["net_surface_fix_only_mm3"], "pin_term_mm3": p["pin_term_mm3"],
                    "footprint_cells": p["footprint"]["footprint_cells"],
                    "footprint_cells_on_window_edge": p["footprint"]["footprint_cells_on_window_edge"],
                    "dirichlet_cells": p["footprint"]["dirichlet_cells"],
                    "components_without_boundary": p["footprint"]["components_without_boundary"],
                    "max_Hpost_reduction_mm": p["Hpost_change_in_footprint_mm"]["max_reduction_mm"],
                    "max_Hpost_raise_mm": p["Hpost_change_in_footprint_mm"]["max_raise_mm"],
                    "corrected_below_Hpre_in_footprint": p["corrected_below_Hpre_in_footprint"],
                    "frozen_below_Hpre_in_footprint": p["frozen_below_Hpre_in_footprint"],
                    "newly_below_Hpre_in_footprint_(frozen_not_below)": nb[D],
                    "coverage_frozen_to_corrected": p["coverage_frozen_to_corrected"],
                    "footprint_cells_in_valid_control_windows": p[
                        "footprint_cells_in_valid_control_windows_(reported_not_applied)"],
                    "per_pin_residual_spur": p["per_pin"],
                    "per_d": {k: {"net_mm3": vv["net_mm3"],
                                  "surface_term_mm3": vv["decomposition_unrounded"]["surface_term_mm3"],
                                  "metal_term_mm3": vv["decomposition_unrounded"]["metal_term_mm3"],
                                  "footprint_cells": vv["footprint"]["footprint_cells"],
                                  "max_Hpost_reduction_mm": vv["Hpost_change_in_footprint_mm"]["max_reduction_mm"],
                                  "corrected_below_Hpre_in_footprint": vv["corrected_below_Hpre_in_footprint"],
                                  "newly_below_Hpre_in_footprint": nb[k]}
                              for k, vv in v.items()}})
    unchanged = {c: all(row["PRIMARY_corrected_net_mm3_d1.4"] == row["frozen_net_mm3"]
                        and all(x == row["frozen_net_mm3"] for x in row["sensitivity_net_mm3"].values())
                        for row in table if row["case"] == c) for c in CASES}
    fin = {"_what": "",
           "date": "", "rule": info,
           "gate": "",
           "boundary_statement": per["S2"]["boundary_statement"],
           "pairs_2x2": {"implant": ["", ""], "no_implant": ["", ""]},
           "net_unchanged_all_d_both_readers": unchanged,
           "S3_note": "",
           "table": table,
           "peak_rss_GiB_apply": {c: per[c]["peak_rss_GiB"] for c in CASES},
           "per_case_files": [f"analysis/{c}/pin_artefact_corrected_{TAG}.json" for c in CASES],
           "script": "analysis/pin_artefact_apply_v2_2026-09-14.py",
           "script_sha256": sha256_file(Path(__file__).resolve())}
    write_new(OUT_FINAL, fin)
    for row in table:
        print(row["case"], row["reader"], row["window"], row["frozen_net_mm3"], "->",
              row["PRIMARY_corrected_net_mm3_d1.4"], row["sensitivity_net_mm3"],
              "surf", row["surface_term_mm3_unrounded"], "metal", row["metal_term_mm3_unrounded"],
              "F", row["footprint_cells"], "maxdrop", row["max_Hpost_reduction_mm"],
              "belowpre", row["corrected_below_Hpre_in_footprint"])


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

