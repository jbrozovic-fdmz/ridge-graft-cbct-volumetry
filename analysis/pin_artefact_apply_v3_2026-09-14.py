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
from pipeline import volume as V

TAG = "v3_2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
RULE3_PY = ROOT / "analysis/pin_artefact_rule_v3_2026-09-14.py"
RULE3_JSON = ROOT / "FINAL_ARCH_20260907/pin_artefact_rule_v3_2026-09-14.json"
RULE3_TEXT_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
RULE3_SCRIPT_SHA = "68410d5e82b3b56b7bdb960f9d5be211ff953903c8316eae9066dc9185db8ff4"
RULE3_PARAMS_SHA = "f3ccfb475c74ae8ebb205967f633e377c8bf021ecc1ba01aa1d4505bb83039ff"
A2_PY = ROOT / "analysis/pin_artefact_apply_v2_2026-09-14.py"
A2_SHA = "6a94880305527b856ffa9ccb3f99b183363f2cf505a29d075d6c5a12c3ffb531"
OUT_FINAL = ROOT / f"FINAL_ARCH_20260907/pin_artefact_corrected_{TAG}.json"
OUT_CMP = ROOT / "FINAL_ARCH_20260907/pin_artefact_attempts_comparison_2026-09-14.json"
QC_RADIUS_MM = 1.0
RESIDUAL_FROZEN_REFERENCE = {("S2", "63"): 9.976, ("S2", "64"): 6.884, ("S4", "760"): 10.864}


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
    assert sha256_file(A2_PY) == A2_SHA, "ABORT: attempt-2 apply script changed"
    A2 = _load("pin_artefact_apply_v2_20260914", A2_PY)
    A1, RULE1, RULE2, rule2, info2 = A2.load_all()
    rule3 = json.loads(RULE3_JSON.read_text())
    assert sha256_file(RULE3_PY) == RULE3_SCRIPT_SHA == rule3["script_sha256_at_declaration"], \
        "ABORT: v3 rule script changed since declaration"
    RULE3 = _load("pin_artefact_rule_v3_20260914", RULE3_PY)
    h = sha256_text(rule3["rule_text"])
    assert h == RULE3_TEXT_SHA == rule3["rule_text_sha256"] == sha256_text(RULE3.RULE_TEXT), "ABORT: v3 rule text"
    ps = sha256_text(json.dumps(RULE3.PARAMS, sort_keys=True))
    assert ps == RULE3_PARAMS_SHA == rule3["parameters_sha256"] and RULE3.PARAMS == rule3["parameters"]
    assert rule3["volumes_loaded"] is False and rule3["corrected_values_computed"] is False
    assert rule3["eligible_pins_per_case"] == info2["eligible_pins_per_case"] == \
        {"S1": [], "S2": [63, 64], "S3": [70], "S4": [441, 735, 760]}
    assert RULE3.PARAMS["d_mm"] == info2["d_mm"] == 1.4 and RULE3.PARAMS["sensitivity_d_mm"] == [1.0, 1.4, 2.0]
    assert RULE3.PARAMS["grid_step_mm"] == A1.GRID_STEP == A1.SWEEP.STEP == 0.20
    assert RULE3.RULE2.footprint.__code__.co_code == RULE2.footprint.__code__.co_code
    for c in CASES:
        for R in READERS:
            assert rule3["frozen_values_to_reproduce_before_correction"][c][R]["net_mm3"] == A1.EXPECTED_NET[c][R]
    assert rule3["frozen_secondary_S4_10x14_to_reproduce"] == A1.EXPECTED_SECONDARY["S4"]
    info = {"rule_json": str(RULE3_JSON.relative_to(ROOT)), "rule_json_file_sha256": sha256_file(RULE3_JSON),
            "rule_text_sha256": h, "rule_text_sha256_verified_json_module": True,
            "rule_script_sha256": RULE3_SCRIPT_SHA, "rule_script_sha256_verified": True,
            "parameters_sha256": ps, "declared_utc": rule3["declared_utc"],
            "d_mm": 1.4, "sensitivity_d_mm": [1.0, 1.4, 2.0],
            "eligible_pins_per_case": rule3["eligible_pins_per_case"],
            "voxel_mm_cap_offset_per_case": rule3["voxel_mm_cap_offset_per_case"],
            "cap_formula_declared": rule3["parameters"]["cap_formula_declared"],
            "reused_attempt2_apply_script": str(A2_PY.relative_to(ROOT)), "reused_attempt2_apply_sha256": A2_SHA,
            "attempt2_rule_text_sha256_verified": info2["rule_text_sha256"],
            "measurement_path_script": info2["measurement_path_script"],
            "measurement_path_script_sha256": info2["measurement_path_script_sha256"]}
    return A1, RULE1, RULE2, A2, RULE3, rule3, info


def gate(case):
    A1, RULE1, RULE2, A2, RULE3, rule3, info = load_all()
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
    print(f"[{case}] v3 gate written; peak RSS {out['peak_rss_GiB']} GiB", flush=True)


def below_pre(Hpre, H, mask):
    with np.errstate(invalid="ignore"):
        d = (H - Hpre)[mask]
    d = d[np.isfinite(d)]
    return {"n_cells_below_Hpre": int((d < 0).sum()), "n_cells_more_than_0.3mm_below": int((d < -0.3).sum()),
            "min_H_minus_Hpre_mm": r(d.min(), 3) if len(d) else None}


def residual_above_titanium(H, near, zt):
    h = H[near]
    h = h[np.isfinite(h)]
    return {"volume_mm3": round(float(np.clip(h - zt, 0, None).sum() * 0.04), 3),
            "cells_above": int((h > zt).sum()),
            "peak_H_minus_titanium_max_Z_mm": round(float(h.max() - zt), 2) if len(h) else None}


def literal_formula_qc(V_, frame, Hpre, Hpost, F, Z, vx, xe, ye, bg, imp, pin_idx):
    Hl = Hpost.copy()
    fin_post, fin_pre = np.isfinite(Hpost), np.isfinite(Hpre)
    cap = Z + float(vx)
    m1 = F & fin_post & fin_pre
    m2 = F & fin_post & ~fin_pre
    Hl[m1] = np.maximum(Hpre[m1], np.minimum(Hpost[m1], cap[m1]))
    Hl[m2] = np.minimum(Hpost[m2], cap[m2])
    raised = m1 & (Hl > Hpost)
    m = V_.measure_window(None, None, None, None, 0.2, background=bg, fields=(Hpre, Hl, xe, ye))
    pin = V_.implant_volume_in_envelope(pin_idx, frame, Hpre, Hl, xe, ye, vx)["implant_in_envelope_mm3"]
    return {"net_mm3": round(m["gain_mm3"] - imp - pin, 1), "cells_raised_vs_frozen": int(raised.sum()),
            "max_raise_mm": r((Hl - Hpost)[raised].max(), 3) if raised.any() else 0.0,
            "_note": ""}


def one_variant(A1, RULE2, RULE3, c, R, w, pin_idx, per_pin, d, implant_term_frozen):
    frame = c.frames[R]; vx = c.vx
    Hpre, Hpost, xe, ye, bg = w["Hpre"], w["Hpost"], w["xe"], w["ye"], w["background"]
    b_pre, b_post = Hpre.tobytes(), Hpost.tobytes()
    o = RULE3.corrected_window(V, frame, Hpre, Hpost, xe, ye, bg, pin_idx, vx, d,
                               implant_term_frozen, w["frozen_net"])
    F, Z, Hc = o["_F"], o["_Zhead"], o["_Hpost_corrected"]
    assert Hpre.tobytes() == b_pre and Hpost.tobytes() == b_post, "frozen inputs modified"
    fin = np.isfinite(Hpost)
    assert np.array_equal(np.isfinite(Hc), fin) and np.all(Hc[fin] <= Hpost[fin]), "P1: a cell increased"
    with np.errstate(invalid="ignore"):
        newly = F & np.isfinite(Hpre) & fin & (Hc < Hpre) & ~(Hpost < Hpre)
    assert not newly.any(), "P2: footprint cell newly below Hpre"
    assert Hc[~F].tobytes() == Hpost[~F].tobytes(), "P3: cell outside footprint changed"
    if pin_idx is not None:
        pxy = frame.to_local(np.asarray(pin_idx, float) * vx)[:, :2]
        assert np.array_equal(F, RULE2.footprint(xe, ye, pxy, d)), "footprint differs from attempt 2"
    g0 = A1.gain_unrounded(Hpre, Hpost, bg)
    g1 = A1.gain_unrounded(Hpre, Hc, bg)
    pin_u = A1.envelope_overlap(pin_idx, frame, Hpre, Hc, xe, ye, vx) if pin_idx is not None else 0.0
    pin_frozen_u = A1.envelope_overlap(pin_idx, frame, Hpre, Hpost, xe, ye, vx) if pin_idx is not None else 0.0
    assert o["pin_term_mm3"] == round(pin_u, 1)
    red = (Hpost - Hc)[F & fin]
    rec = {"d_mm": d, "net_mm3": o["net_mm3"], "frozen_net_mm3": w["frozen_net"],
           "delta_net_mm3": round(o["net_mm3"] - w["frozen_net"], 1),
           "delta_pct": round(100.0 * (o["net_mm3"] - w["frozen_net"]) / w["frozen_net"], 1),
           "net_cap_only_mm3": o["net_cap_only_mm3"], "gross_mm3": o["gross_mm3"],
           "frozen_gross_mm3": o["frozen_gross_mm3"], "implant_term_frozen_mm3": implant_term_frozen,
           "pin_term_mm3": o["pin_term_mm3"], "coverage": o["coverage"],
           "coverage_frozen": w["frozen_coverage"], "area_mm2": o["area_mm2"],
           "decomposition_rounded": {"cap_surface_term_mm3": round(o["net_cap_only_mm3"] - w["frozen_net"], 1),
                                     "metal_term_mm3": round(-o["pin_term_mm3"], 1)},
           "decomposition_unrounded": {"cap_surface_term_mm3": r(g1 - g0, 3), "metal_term_mm3": r(-pin_u, 3),
                                       "delta_net_mm3": r((g1 - g0) - pin_u, 3),
                                       "background_term_mm3": 0.0, "implant_term_change_mm3": 0.0,
                                       "eligible_pin_metal_on_FROZEN_fields_mm3_(reference,_not_additive)":
                                           r(pin_frozen_u, 3)},
           "cap": dict(o["cap"]),
           "cells_lowered": o["cap"]["cells_capped_lowered"],
           "max_lowering_mm": r(o["cap"]["max_Hpost_reduction_mm"], 3),
           "cells_newly_below_Hpre_asserted_zero": int(newly.sum()),
           "frozen_below_Hpre_in_footprint": below_pre(Hpre, Hpost, F),
           "corrected_below_Hpre_in_footprint": below_pre(Hpre, Hc, F),
           "footprint_cells_in_valid_control_windows_(reported_not_applied)": {
               nm: int(RULE2.footprint(V._edges(*xr, 0.2), V._edges(*yr, 0.2),
                                       frame.to_local(np.asarray(pin_idx, float) * vx)[:, :2], d).sum())
               if pin_idx is not None else 0
               for nm, (xr, yr) in w["controls"].items()},
           "per_pin": {}}
    if w["implant_override"] is None and c.implant_idx is not None:
        rec["implant_term_recomputed_on_Hc_QC_not_applied_mm3"] = V.implant_volume_in_envelope(
            c.implant_idx, frame, Hpre, Hc, xe, ye, vx)["implant_in_envelope_mm3"]
    if pin_idx is not None:
        rec["QC_literal_D7d_formula_NOT_DECLARED"] = literal_formula_qc(
            V, frame, Hpre, Hpost, F, Z, vx, xe, ye, bg, implant_term_frozen, pin_idx)
    if pin_idx is not None and F.any():
        capv = (Z + vx)[F]
        rec["cap_level_minus_Hpre_in_footprint_mm_min_median_max"] = [
            r(np.nanmin((Z + vx - Hpre)[F]), 3), r(np.nanmedian((Z + vx - Hpre)[F]), 3),
            r(np.nanmax((Z + vx - Hpre)[F]), 3)] if np.isfinite(Hpre[F]).any() else None
        rec["cap_level_range_mm"] = [r(capv.min(), 3), r(capv.max(), 3)]
    for lb, (xy, zmax) in per_pin.items():
        near = RULE2.footprint(xe, ye, xy, QC_RADIUS_MM)
        Fp = RULE2.footprint(xe, ye, xy, d)
        rec["per_pin"][str(lb)] = {
            "footprint_cells_this_pin": int(Fp.sum()), "cells_within_1.0mm": int(near.sum()),
            "titanium_max_Z_mm": r(zmax, 2),
            "residual_above_titanium_1mm_FROZEN": residual_above_titanium(Hpost, near, zmax) if near.any() else None,
            "residual_above_titanium_1mm_CORRECTED": residual_above_titanium(Hc, near, zmax) if near.any() else None,
            "cells_lowered_within_1.0mm": int((near & fin & (Hc < Hpost)).sum())}
    return rec, F, Z, Hc


def apply(case):
    A1, RULE1, RULE2, A2, RULE3, rule3, info = load_all()
    gates = {}
    for cc in CASES:
        gp = ROOT / f"analysis/{cc}/pin_artefact_gate_{TAG}.json"
        assert gp.exists(), f"ABORT: v3 gate file missing for {cc}; run --gate for all four cases first"
        g = json.loads(gp.read_text())
        assert g["all_assertions_passed"] is True and g["rule"]["rule_text_sha256"] == info["rule_text_sha256"]
        for R in READERS:
            assert g["readers"][R]["net_mm3"] == A1.EXPECTED_NET[cc][R]
            if cc in A1.EXPECTED_SECONDARY:
                assert g["readers"][R]["secondary"]["net_mm3"] == A1.EXPECTED_SECONDARY[cc][R]
        gates[cc] = {R: g["readers"][R]["net_mm3"] for R in READERS}
        if cc in A1.EXPECTED_SECONDARY:
            gates[cc]["secondary_10x14"] = {R: g["readers"][R]["secondary"]["net_mm3"] for R in READERS}
    p = ROOT / f"analysis/{case}/pin_artefact_corrected_{TAG}.json"
    pz = ROOT / f"analysis/{case}/pin_artefact_fields_{TAG}.npz"
    for q in (p, pz):
        if q.exists():
            raise SystemExit(f"{q} exists; not overwritten")
    t0 = time.time()
    c = A1.setup(case, RULE1)
    assert c.vx == rule3["voxel_mm_cap_offset_per_case"][case], (c.vx, rule3["voxel_mm_cap_offset_per_case"][case])
    Q = A1.post_clouds(c, c.post)
    del c.post
    base = {}
    for R in READERS:
        base[R] = A1.measure_reader(c, R, Q[R], None)
        A1.assert_reproduces(c, R, base[R])
    del Q
    elig = info["eligible_pins_per_case"][case]
    decl_rows = {row["label"]: row for row in rule3["pin_candidates_with_eligibility"][case]}
    got_rows = {row["label"]: row for row in c.pin_rows}
    assert set(got_rows) == set(decl_rows), (got_rows.keys(), decl_rows.keys())
    for lb in elig:
        assert decl_rows[lb]["eligible"] and got_rows[lb]["n_voxels"] == decl_rows[lb]["n_voxels"] \
            and got_rows[lb]["post_grey_max"] == round(decl_rows[lb]["post_grey_max"]), lb
    pin_vox = {lb: np.argwhere(c.lbl == lb) + c.lo for lb in elig}
    pin_idx = np.vstack([pin_vox[lb] for lb in elig]) if elig else None
    out = {"case": case, "_what": "",
           "rule": info, "gate_frozen_values_reproduced_v3": gates, "pins_all_candidates": c.pin_rows,
           "eligible_pins": elig, "eligible_pin_voxels": {str(lb): int(len(v)) for lb, v in pin_vox.items()},
           "voxel_mm_cap_offset": c.vx, "implant_label": c.implant_label,
           "mem_available_GiB_at_load": c.mem_available_GiB_at_load,
           "reproduced_unmodified": {R: A1.public(base[R]) for R in READERS}, "readers": {}}
    npz = {}
    for R in READERS:
        frame = c.frames[R]
        per_pin = {}
        for lb in elig:
            L = frame.to_local(pin_vox[lb].astype(float) * c.vx)
            per_pin[lb] = (L[:, :2], float(L[:, 2].max()))
        wins = A2.windows_for(c, R, base[R], A1)
        out["readers"][R] = {}
        for wn, w in wins.items():
            implant_term_frozen = (base[R]["implant_in_envelope_mm3"] if w["implant_override"] is None
                                   else float(w["implant_override"]))
            idt = RULE3.corrected_window(V, frame, w["Hpre"], w["Hpost"], w["xe"], w["ye"], w["background"],
                                         None, c.vx, info["d_mm"], implant_term_frozen, w["frozen_net"])
            assert idt["net_mm3"] == w["frozen_net"] and idt["_Hpost_corrected"].tobytes() == w["Hpost"].tobytes()
            wrec = {"grid_shape": list(w["Hpre"].shape), "X_mm": [r(w["xe"][0], 2), r(w["xe"][-1], 2)],
                    "Y_mm": [r(w["ye"][0], 2), r(w["ye"][-1], 2)], "background_frozen_mm": r(w["background"], 4),
                    "frozen_net_mm3": w["frozen_net"], "implant_term_frozen_mm3": implant_term_frozen,
                    "identity_no_pin_reproduces_frozen": True, "variants": {}}
            key0 = f"{R}_{wn}"
            npz[f"{key0}_Hpre"] = w["Hpre"]; npz[f"{key0}_Hpost"] = w["Hpost"]
            npz[f"{key0}_xe"] = w["xe"]; npz[f"{key0}_ye"] = w["ye"]
            for d in info["sensitivity_d_mm"]:
                rec, F, Z, Hc = one_variant(A1, RULE2, RULE3, c, R, w, pin_idx, per_pin, d, implant_term_frozen)
                if pin_idx is None:
                    assert rec["net_mm3"] == w["frozen_net"] and Hc.tobytes() == w["Hpost"].tobytes()
                    rec["note"] = ""
                rec["role"] = "PRIMARY" if d == info["d_mm"] else "sensitivity"
                if wn == "primary_10x10" and R == "A":
                    for (cs_, lb), ref in RESIDUAL_FROZEN_REFERENCE.items():
                        if cs_ == case:
                            got = rec["per_pin"][lb]["residual_above_titanium_1mm_FROZEN"]["volume_mm3"]
                            assert got == ref, ("residual metric does not reproduce the verifier", lb, got, ref)
                            rec["per_pin"][lb]["frozen_residual_reproduces_verifier_value"] = ref
                wrec["variants"][f"d={d:.1f}"] = rec
                npz[f"{key0}_d{d:.1f}_F"] = F; npz[f"{key0}_d{d:.1f}_Zhead"] = Z; npz[f"{key0}_d{d:.1f}_Hc"] = Hc
                print(f"  {case}/{R} {wn} d={d:.1f}: net {rec['net_mm3']} (frozen {w['frozen_net']}; cap "
                      f"{rec['decomposition_unrounded']['cap_surface_term_mm3']}, metal "
                      f"{rec['decomposition_unrounded']['metal_term_mm3']}); F {rec['cap']['footprint_cells']}; "
                      f"lowered {rec['cells_lowered']}; max {rec['max_lowering_mm']}", flush=True)
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
    A1, RULE1, RULE2, A2, RULE3, rule3, info = load_all()
    per = {}
    for c in CASES:
        g = json.loads((ROOT / f"analysis/{c}/pin_artefact_gate_{TAG}.json").read_text())
        a = json.loads((ROOT / f"analysis/{c}/pin_artefact_corrected_{TAG}.json").read_text())
        assert g["all_assertions_passed"] and a["rule"]["rule_text_sha256"] == info["rule_text_sha256"]
        assert sha256_file(ROOT / a["fields_npz"]) == a["fields_npz_sha256"]
        per[c] = a
    D = "d=1.4"
    table = []
    for c in CASES:
        a = per[c]
        for R in READERS:
            for wn, w in a["readers"][R].items():
                v = w["variants"]; p = v[D]
                table.append({
                    "case": c, "reader": R, "window": wn, "eligible_pins": a["eligible_pins"],
                    "frozen_net_mm3": w["frozen_net_mm3"],
                    "PRIMARY_corrected_net_mm3_d1.4": p["net_mm3"], "delta_mm3": p["delta_net_mm3"],
                    "delta_pct": p["delta_pct"],
                    "sensitivity_net_mm3": {k: vv["net_mm3"] for k, vv in v.items()},
                    "cap_surface_term_mm3_unrounded": p["decomposition_unrounded"]["cap_surface_term_mm3"],
                    "metal_term_mm3_unrounded": p["decomposition_unrounded"]["metal_term_mm3"],
                    "cap_surface_term_mm3_rounded": p["decomposition_rounded"]["cap_surface_term_mm3"],
                    "metal_term_mm3_rounded": p["decomposition_rounded"]["metal_term_mm3"],
                    "net_cap_only_mm3": p["net_cap_only_mm3"], "pin_term_mm3": p["pin_term_mm3"],
                    "implant_term_frozen_mm3": p["implant_term_frozen_mm3"],
                    "footprint_cells": p["cap"]["footprint_cells"], "cells_lowered": p["cells_lowered"],
                    "max_lowering_mm": p["max_lowering_mm"],
                    "cells_floored_at_Hpre": p["cap"]["cells_floored_at_Hpre"],
                    "frozen_below_Hpre_cells_left_unchanged": p["cap"]["frozen_below_Hpre_cells_left_unchanged"],
                    "Hpre_nonfinite_cells_capped": p["cap"]["Hpre_nonfinite_cells_capped"],
                    "cells_newly_below_Hpre": p["cells_newly_below_Hpre_asserted_zero"],
                    "coverage_frozen_corrected": [p["coverage_frozen"], p["coverage"]],
                    "implant_term_recomputed_on_Hc_QC_not_applied_mm3":
                        p.get("implant_term_recomputed_on_Hc_QC_not_applied_mm3"),
                    "footprint_cells_in_valid_control_windows":
                        p["footprint_cells_in_valid_control_windows_(reported_not_applied)"],
                    "QC_literal_D7d_formula_NOT_DECLARED": p.get("QC_literal_D7d_formula_NOT_DECLARED"),
                    "per_pin_residual_above_titanium_1mm": {
                        lb: {"frozen": q["residual_above_titanium_1mm_FROZEN"],
                             "corrected": q["residual_above_titanium_1mm_CORRECTED"],
                             "footprint_cells_this_pin": q["footprint_cells_this_pin"]}
                        for lb, q in p["per_pin"].items()},
                    "per_d": {k: {"net_mm3": vv["net_mm3"],
                                  "cap_surface_term_mm3": vv["decomposition_unrounded"]["cap_surface_term_mm3"],
                                  "metal_term_mm3": vv["decomposition_unrounded"]["metal_term_mm3"],
                                  "footprint_cells": vv["cap"]["footprint_cells"],
                                  "cells_lowered": vv["cells_lowered"], "max_lowering_mm": vv["max_lowering_mm"],
                                  "per_pin_residual_corrected_mm3": {
                                      lb: (q["residual_above_titanium_1mm_CORRECTED"] or {}).get("volume_mm3")
                                      for lb, q in vv["per_pin"].items()},
                                  "QC_literal_formula_net_mm3": (vv.get("QC_literal_D7d_formula_NOT_DECLARED")
                                                                 or {}).get("net_mm3")}
                              for k, vv in v.items()}})
    fin = {"_what": "",
           "date": "", "rule": info,
           "gate": "",
           "properties_asserted_every_variant": ["",
                                                 "",
                                                 "",
                                                 ""],
           "cap_formula_note": "",
           "residual_metric": "",
           "table": table,
           "peak_rss_GiB_apply": {c: per[c]["peak_rss_GiB"] for c in CASES},
           "per_case_files": [f"analysis/{c}/pin_artefact_corrected_{TAG}.json" for c in CASES],
           "script": f"analysis/pin_artefact_apply_{TAG}.py", "script_sha256": sha256_file(Path(__file__).resolve())}
    write_new(OUT_FINAL, fin)
    for row in table:
        print(row["case"], row["reader"], row["window"], row["frozen_net_mm3"], "->",
              row["PRIMARY_corrected_net_mm3_d1.4"], row["sensitivity_net_mm3"],
              "cap", row["cap_surface_term_mm3_unrounded"], "metal", row["metal_term_mm3_unrounded"],
              "F", row["footprint_cells"], "lowered", row["cells_lowered"], "max", row["max_lowering_mm"],
              {k: (v["frozen"] or {}).get("volume_mm3") for k, v in row["per_pin_residual_above_titanium_1mm"].items()},
              {k: (v["corrected"] or {}).get("volume_mm3") for k, v in row["per_pin_residual_above_titanium_1mm"].items()},
              "literal", (row["QC_literal_D7d_formula_NOT_DECLARED"] or {}).get("net_mm3"))


def compare():
    src = {"metal_only_2026-09-13": ROOT / "FINAL_ARCH_20260907/fixation_pins_2026-09-13.json",
           "attempt1": ROOT / "FINAL_ARCH_20260907/pin_artefact_corrected_2026-09-14.json",
           "attempt2": ROOT / "FINAL_ARCH_20260907/pin_artefact_corrected_v2_2026-09-14.json",
           "attempt3": OUT_FINAL}
    J = {k: json.loads(p.read_text()) for k, p in src.items()}
    a3 = {(t["case"], t["reader"], t["window"]): t for t in J["attempt3"]["table"]}
    a2 = {(t["case"], t["reader"], t["window"]): t for t in J["attempt2"]["table"]}
    a1 = {(t["case"], t["reader"]): t for t in J["attempt1"]["table"]}
    m0 = {(t["case"], t["reader"]): t for t in J["metal_only_2026-09-13"]["table"]
          if t["mask_set"] == "pin_candidates" and t["margin_mm"] == 0.0}
    rows = []
    for key in sorted(a3, key=lambda k: (k[0], k[1], k[2] != "primary_10x10")):
        c, R, wn = key
        t3, t2, t1, t0 = a3[key], a2[key], a1[(c, R)], m0[(c, R)]
        prim = wn == "primary_10x10"
        if prim:
            frozen = t3["frozen_net_mm3"]
            assert frozen == t2["frozen_net_mm3"] == t1["frozen_net_mm3"] == t0["frozen_net_mm3"]
            mo = t0["subtraction_net_mm3"]
            a1v = {"d=1.1": t1["net_d1.1_mm3"], "d=1.4": t1["net_d1.4_mm3"], "d=1.7": t1["net_d1.7_mm3"]}
        else:
            frozen = t3["frozen_net_mm3"]
            s1 = t1["secondary_10x14"]
            assert frozen == t2["frozen_net_mm3"] == s1["frozen_mm3"] == t0["secondary_frozen_mm3"]
            mo = t0["secondary_subtraction_net_mm3"]
            a1v = {"d=1.1": s1["net_d1.1_mm3"], "d=1.4": s1["net_d1.4_mm3"], "d=1.7": s1["net_d1.7_mm3"]}
        rows.append({"case": c, "reader": R, "window": wn, "frozen_net_mm3": frozen,
                     "metal_only_2026-09-13_mm3": mo,
                     "attempt1_voxel_inpaint_d1.4_mm3": a1v["d=1.4"], "attempt1_sensitivity_mm3": a1v,
                     "attempt2_harmonic_fill_d1.4_mm3": t2["PRIMARY_corrected_net_mm3_d1.4"],
                     "attempt2_sensitivity_mm3": t2["sensitivity_net_mm3"],
                     "attempt3_pin_head_cap_d1.4_mm3": t3["PRIMARY_corrected_net_mm3_d1.4"],
                     "attempt3_sensitivity_mm3": t3["sensitivity_net_mm3"]})
    out = {"_what": "",
           "date": "",
           "columns": {
               "metal_only_2026-09-13_mm3": "",
               "attempt1": "",
               "attempt2": "",
               "attempt3": ""},
           "sources": {k: {"file": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for k, p in src.items()},
           "table": rows, "script": f"analysis/pin_artefact_apply_{TAG}.py",
           "script_sha256": sha256_file(Path(__file__).resolve())}
    write_new(OUT_CMP, out)
    for x in rows:
        print(x["case"], x["reader"], x["window"], x["frozen_net_mm3"], x["metal_only_2026-09-13_mm3"],
              x["attempt1_voxel_inpaint_d1.4_mm3"], x["attempt2_harmonic_fill_d1.4_mm3"],
              x["attempt3_pin_head_cap_d1.4_mm3"], x["attempt3_sensitivity_mm3"])


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--gate" and a[1] in CASES:
        gate(a[1])
    elif len(a) == 2 and a[0] == "--apply" and a[1] in CASES:
        apply(a[1])
    elif a == ["--collect"]:
        collect()
    elif a == ["--compare"]:
        compare()
    else:
        raise SystemExit(__doc__)

