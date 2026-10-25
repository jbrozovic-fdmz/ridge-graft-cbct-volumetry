#!/usr/bin/env python3
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.spatial import cKDTree

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "analysis"))
TAG = "v3_2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
OUT = ROOT / f"FINAL_ARCH_20260907/pin_artefact_rule_{TAG}.json"
RULE2_PY = ROOT / "analysis/pin_artefact_rule_v2_2026-09-14.py"
RULE2_JSON = ROOT / "FINAL_ARCH_20260907/pin_artefact_rule_v2_2026-09-14.json"
RULE2_SCRIPT_SHA = "845a610d0aec21f28f654d02a559dd15bd934187501c676b0e8b1ae6f408e8d3"
RULE2_TEXT_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
A2_APPLY_PY = ROOT / "analysis/pin_artefact_apply_v2_2026-09-14.py"
A1_APPLY_PY = ROOT / "analysis/pin_artefact_apply_2026-09-14.py"
A1_APPLY_SHA = "552187e484a8d11a59cf9cb7190c930bd0c6488da30c8c6eb33748a8389d0875"
FIG2_PY = ROOT / "analysis/fig_pin_heads_CORRECTED_v2_2026-09-14.py"

PARAMS = {
    "eligibility": "",
    "d_mm": 1.4,
    "sensitivity_d_mm": [1.0, 1.4, 2.0],
    "distance_tolerance_mm": 1e-6,
    "footprint": "",
    "zhead": "",
    "cap_offset": "",
    "cap_formula_declared": "",
    "cap_formula_Hpre_nonfinite": "",
    "cap_formula_Hpost_nonfinite": "unchanged (NaN)",
    "grid_step_mm": 0.20,
    "windows": {"primary": "",
                "secondary_S4": ""},
    "metal_term": "",
    "implant_term": "",
    "background": "",
    "net": "",
}

RULE_TEXT = ""


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p: Path, obj) -> None:
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def load_rule2():
    assert sha256_file(RULE2_PY) == RULE2_SCRIPT_SHA, "ABORT: rule v2 script changed since its declaration"
    spec = importlib.util.spec_from_file_location("pin_artefact_rule_v2_20260914", RULE2_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert sha256_text(m.RULE_TEXT) == RULE2_TEXT_SHA, "ABORT: rule v2 text changed"
    assert m.PARAMS["d_mm"] == PARAMS["d_mm"] and m.PARAMS["sensitivity_d_mm"] == PARAMS["sensitivity_d_mm"]
    assert m.PARAMS["distance_tolerance_mm"] == PARAMS["distance_tolerance_mm"]
    assert m.PARAMS["grid_step_mm"] == PARAMS["grid_step_mm"]
    return m


RULE2 = load_rule2()


def zhead_field(xe, ye, pin_xyz_local, d):
    xe = np.asarray(xe, float); ye = np.asarray(ye, float)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    ny, nx = len(yc), len(xc)
    Z = np.full((ny, nx), np.nan)
    if pin_xyz_local is None or not len(pin_xyz_local):
        return np.zeros((ny, nx), bool), Z
    P = np.asarray(pin_xyz_local, float)
    F = RULE2.footprint(xe, ye, P[:, :2], d)
    r = d + PARAMS["distance_tolerance_mm"]
    tree = cKDTree(P[:, :2])
    X, Y = np.meshgrid(xc, yc)
    G = np.column_stack([X.ravel(), Y.ravel()])
    cand = tree.query_ball_point(G, r + 1e-3)
    Zr = Z.ravel()
    for k, lst in enumerate(cand):
        if not lst:
            continue
        lst = np.asarray(lst, np.int64)
        dist = np.hypot(P[lst, 0] - G[k, 0], P[lst, 1] - G[k, 1])
        sel = lst[dist <= r]
        if len(sel):
            Zr[k] = P[sel, 2].max()
    Z = Zr.reshape(ny, nx)
    assert np.array_equal(np.isfinite(Z), F), "ABORT: head-height support differs from the v2 footprint"
    return F, Z


def assert_cell_properties(Hpre, Hpost, Hc, F):
    Hpre = np.asarray(Hpre); Hpost = np.asarray(Hpost); Hc = np.asarray(Hc); F = np.asarray(F, bool)
    assert Hc.shape == Hpost.shape == Hpre.shape == F.shape and Hc.dtype == Hpost.dtype
    fin = np.isfinite(Hpost)
    assert np.array_equal(np.isfinite(Hc), fin), "P1 FAIL: finite-cell pattern changed"
    assert np.all(Hc[fin] <= Hpost[fin]), "P1 FAIL: a cell increased"
    both = F & np.isfinite(Hpre) & fin
    newly = both & (Hc < Hpre) & ~(Hpost < Hpre)
    assert not newly.any(), f"P2 FAIL: {int(newly.sum())} footprint cells newly below Hpre"
    assert Hc[~F].tobytes() == Hpost[~F].tobytes(), "P3 FAIL: a cell outside the footprint changed"
    return True


def cap_surface(Hpre, Hpost, F, Zhead, voxel_mm):
    Hpre = np.asarray(Hpre); Hpost = np.asarray(Hpost); F = np.asarray(F, bool)
    b_pre, b_post = Hpre.tobytes(), Hpost.tobytes()
    Hc = Hpost.copy()
    fin_post = np.isfinite(Hpost); fin_pre = np.isfinite(Hpre)
    assert np.all(np.isfinite(Zhead[F])), "Zhead must be finite on the footprint"
    cap = Zhead + float(voxel_mm)
    m1 = F & fin_post & fin_pre
    m2 = F & fin_post & ~fin_pre
    Hc[m1] = np.minimum(Hpost[m1], np.maximum(Hpre[m1], cap[m1]))
    Hc[m2] = np.minimum(Hpost[m2], cap[m2])
    assert_cell_properties(Hpre, Hpost, Hc, F)
    assert Hpre.tobytes() == b_pre and Hpost.tobytes() == b_post, "P3 FAIL: an input array was modified"
    red = (Hpost - Hc)[F & fin_post]
    info = {"footprint_cells": int(F.sum()),
            "footprint_cells_finite_Hpost": int((F & fin_post).sum()),
            "cells_capped_lowered": int((red > 0).sum()),
            "cells_floored_at_Hpre": int((m1 & (cap < Hpre) & (Hpost > Hpre)).sum()),
            "frozen_below_Hpre_cells_left_unchanged": int((m1 & (Hpost < Hpre)).sum()),
            "Hpre_nonfinite_cells_capped": int((m2 & (Hc < Hpost)).sum()),
            "max_Hpost_reduction_mm": float(red.max()) if len(red) else 0.0,
            "sum_Hpost_reduction_mm": float(red.sum()) if len(red) else 0.0,
            "cells_newly_below_Hpre": 0,
            "properties_P1_P2_P3_asserted": True}
    return Hc, info


def corrected_window(V, frame, Hpre, Hpost, xe, ye, background, pin_idx_SAL, vx, d,
                     implant_term_frozen, frozen_net):
    step = PARAMS["grid_step_mm"]
    m0 = V.measure_window(None, None, None, None, step, background=background, fields=(Hpre, Hpost, xe, ye))
    assert round(m0["gain_mm3"] - implant_term_frozen, 1) == frozen_net, "frozen identity failed"
    pin_idx = None if pin_idx_SAL is None or not len(pin_idx_SAL) else np.asarray(pin_idx_SAL)
    if pin_idx is None:
        F, Z = zhead_field(xe, ye, None, d)
    else:
        L = frame.to_local(pin_idx.astype(float) * vx)
        F, Z = zhead_field(xe, ye, L, d)
    Hc, info = cap_surface(Hpre, Hpost, F, Z, vx)
    m = V.measure_window(None, None, None, None, step, background=background, fields=(Hpre, Hc, xe, ye))
    pin = (V.implant_volume_in_envelope(pin_idx, frame, Hpre, Hc, xe, ye, vx)["implant_in_envelope_mm3"]
           if pin_idx is not None else 0.0)
    net = round(m["gain_mm3"] - implant_term_frozen - pin, 1)
    assert m["coverage"] == m0["coverage"] and m["cells"] == m0["cells"], "coverage changed"
    assert m["gain_mm3"] <= m0["gain_mm3"], "gross increased"
    assert net <= frozen_net, "net increased"
    if pin_idx is None:
        assert net == frozen_net and Hc.tobytes() == Hpost.tobytes()
    return {"net_mm3": net, "gross_mm3": m["gain_mm3"], "frozen_gross_mm3": m0["gain_mm3"],
            "implant_term_frozen_mm3": implant_term_frozen, "pin_term_mm3": pin,
            "net_cap_only_mm3": round(m["gain_mm3"] - implant_term_frozen, 1),
            "coverage": m["coverage"], "area_mm2": m["area_mm2"], "cap": info,
            "_F": F, "_Zhead": Z, "_Hpost_corrected": Hc}


def selftest():
    from pipeline import geom, volume as V
    res = {}
    rng = np.random.default_rng(20260914)
    step = PARAMS["grid_step_mm"]
    xe = np.arange(10.0, 20.0 + 1e-9, step); ye = np.arange(15.0, 25.0 + 1e-9, step)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    ny, nx = X.shape
    vx = 0.30
    pins = []
    for (px, py, ztop) in ((12.9, 22.9, 21.4), (17.3, 23.6, 21.6)):
        for dz in np.arange(0, 3.0, vx):
            for ox in (-0.3, 0.0, 0.3):
                for oy in (-0.3, 0.0, 0.3):
                    pins.append((px + ox, py + oy, ztop - dz))
    P = np.array(pins)
    Hpre = 19.0 + 0.05 * (X - 10) + rng.normal(0, 0.02, X.shape)
    Hpost = Hpre + 1.5 + rng.normal(0, 0.05, X.shape)
    bump = np.exp(-((X - 12.9) ** 2 + (Y - 22.9) ** 2) / 0.8) * 4.0 + np.exp(-((X - 17.3) ** 2 + (Y - 23.6) ** 2) / 0.8) * 3.5
    Hpost = Hpost + bump
    Hpost[5, 5] = np.nan; Hpre[7, 7] = np.nan
    Hpost[41, 15] = np.nan
    Hpre[39, 14] = np.nan
    Hpost[40, 16] = Hpre[40, 16] - 0.6
    Hpre[36, 12] = 21.9; Hpost[36, 12] = 23.5
    Hpre0, Hpost0 = Hpre.copy(), Hpost.copy()

    for d in PARAMS["sensitivity_d_mm"]:
        F, Z = zhead_field(xe, ye, P, d)
        Dall = np.hypot(X[..., None] - P[:, 0], Y[..., None] - P[:, 1])
        within = Dall <= d + PARAMS["distance_tolerance_mm"]
        Fb = within.any(-1)
        Zb = np.where(Fb, np.where(within, P[:, 2], -np.inf).max(-1), np.nan)
        assert np.array_equal(F, Fb) and np.array_equal(F, RULE2.footprint(xe, ye, P[:, :2], d))
        assert np.array_equal(Z, Zb, equal_nan=True)
        Hc, info = cap_surface(Hpre, Hpost, F, Z, vx)
        cap = Z + vx
        m = F & np.isfinite(Hpre) & np.isfinite(Hpost) & (Hpost >= Hpre)
        lit = np.maximum(Hpre, np.minimum(Hpost, cap))
        assert np.array_equal(Hc[m], lit[m])
        mb = F & np.isfinite(Hpre) & np.isfinite(Hpost) & (Hpost < Hpre)
        assert mb.any() and np.array_equal(Hc[mb], Hpost[mb])
        assert np.all(lit[mb] > Hpost[mb])
        fin = np.isfinite(Hpost)
        p1 = bool(np.array_equal(np.isfinite(Hc), fin) and np.all(Hc[fin] <= Hpost[fin]))
        both = F & np.isfinite(Hpre) & fin
        p2 = bool(not (both & (Hc < Hpre) & ~(Hpost < Hpre)).any())
        p3 = bool(Hc[~F].tobytes() == Hpost[~F].tobytes())
        assert p1 and p2 and p3
        assert Hpre.tobytes() == Hpre0.tobytes() and Hpost.tobytes() == Hpost0.tobytes()
        ff = F & np.isfinite(Hc)
        assert np.all(Hc[ff] <= np.fmax(Hpre[ff], cap[ff]))
        assert info["cells_capped_lowered"] > 0 and info["cells_floored_at_Hpre"] >= 1
        assert info["Hpre_nonfinite_cells_capped"] >= (1 if F[39, 14] else 0)
        Hc2, _ = cap_surface(Hpre, Hc, F, Z, vx)
        assert Hc2.tobytes() == Hc.tobytes()
        assert cap_surface(Hpre, Hpost, F, Z, vx)[0].tobytes() == Hc.tobytes()
        res[f"d={d:.1f}"] = {"P1_no_cell_increases": p1, "P2_no_footprint_cell_newly_below_Hpre": p2,
                             "P3_outside_footprint_bit_identical": p3, "footprint_cells": info["footprint_cells"],
                             "cells_capped_lowered": info["cells_capped_lowered"],
                             "cells_floored_at_Hpre": info["cells_floored_at_Hpre"],
                             "frozen_below_cells_left_unchanged_literal_would_raise": int(mb.sum()),
                             "Hpre_nonfinite_cells_capped": info["Hpre_nonfinite_cells_capped"],
                             "equals_literal_formula_where_Hpost_ge_Hpre": True,
                             "idempotent": True}

    F, Z = zhead_field(xe, ye, P, 1.4)
    lit_all = Hpost.copy()
    mm = F & np.isfinite(Hpre) & np.isfinite(Hpost)
    lit_all[mm] = np.maximum(Hpre[mm], np.minimum(Hpost[mm], (Z + vx)[mm]))
    neg = {}
    for name, bad, expect in (
            ("literal_max_formula_raises_frozen_below", lit_all, "P1"),
            ("cell_outside_footprint_lowered", None, "P3"),
            ("footprint_cell_pushed_below_Hpre", None, "P2"),
            ("nan_filled", None, "P1")):
        if bad is None:
            bad = cap_surface(Hpre, Hpost, F, Z, vx)[0].copy()
            if expect == "P3":
                i, j = np.argwhere(~F & np.isfinite(Hpost))[0]; bad[i, j] -= 1e-9
            elif expect == "P2":
                i, j = np.argwhere(F & np.isfinite(Hpre) & np.isfinite(Hpost) & (Hpost > Hpre + 0.5))[0]
                bad[i, j] = Hpre[i, j] - 0.1
            else:
                bad[5, 5] = 1.0
        try:
            assert_cell_properties(Hpre, Hpost, bad, F)
            neg[name] = "NOT CAUGHT"
        except AssertionError as e:
            neg[name] = str(e).split(":")[0]
        assert neg[name] == f"{expect} FAIL", (name, neg[name])
    res["negative_tests_caught"] = neg

    frame = geom.LocalFrame(origin=[0, 0, 0], e_md=[1, 0, 0], e_apical=[0, 1, 0], e_buccal=[0, 0, 1],
                            offset=[0, 0, 0])
    idx = np.rint(P / vx).astype(np.int64)
    Pl = frame.to_local(idx.astype(float) * vx)
    bg = -0.05
    m0 = V.measure_window(None, None, None, None, step, background=bg, fields=(Hpre, Hpost, xe, ye))
    frozen_net = round(m0["gain_mm3"] - 0.0, 1)
    w = {}
    for d in PARAMS["sensitivity_d_mm"]:
        o = corrected_window(V, frame, Hpre, Hpost, xe, ye, bg, idx, vx, d, 0.0, frozen_net)
        Fd, Zd = zhead_field(xe, ye, Pl, d)
        assert np.array_equal(o["_F"], Fd)
        ref_pin = V.implant_volume_in_envelope(idx, frame, Hpre, o["_Hpost_corrected"], xe, ye, vx)
        assert o["pin_term_mm3"] == ref_pin["implant_in_envelope_mm3"]
        w[f"d={d:.1f}"] = {"frozen_net": frozen_net, "net": o["net_mm3"], "pin_term": o["pin_term_mm3"],
                           "coverage_unchanged": o["coverage"] == m0["coverage"],
                           "net_le_frozen": o["net_mm3"] <= frozen_net}
    idt = corrected_window(V, frame, Hpre, Hpost, xe, ye, bg, None, vx, 1.4, 0.0, frozen_net)
    assert idt["net_mm3"] == frozen_net and idt["_Hpost_corrected"].tobytes() == Hpost.tobytes()
    w["no_pin_identity_exact"] = True
    res["window_level_synthetic"] = w
    res["all_passed"] = True
    return res


def declare():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwritten")
    v3_outputs = [p for p in glob.glob(str(ROOT / f"analysis/S*/*{TAG}*"))
                  + glob.glob(str(ROOT / f"FINAL_ARCH_20260907/*{TAG}*"))
                  + glob.glob(str(ROOT / f"JFB_REVISION_2026-09-13/**/*{TAG}*"), recursive=True)]
    assert not v3_outputs, ("attempt-3 outputs already exist; declaration would not precede them", v3_outputs)
    st = selftest()
    rule2 = json.loads(RULE2_JSON.read_text())
    assert rule2["rule_text_sha256"] == RULE2_TEXT_SHA and rule2["script_sha256_at_declaration"] == RULE2_SCRIPT_SHA
    elig, rows = RULE2.eligible_pins()
    assert elig == rule2["eligible_pins_per_case"] == {"S1": [], "S2": [63, 64], "S3": [70], "S4": [441, 735, 760]}
    assert sha256_file(A1_APPLY_PY) == A1_APPLY_SHA
    frz = json.loads((ROOT / "FINAL_ARCH_20260907/FROZEN_RESULTS.json").read_text())["cases"]
    expected = {"S1": {"A": 105.6, "B": 104.8}, "S2": {"A": 119.9, "B": 108.0},
                "S3": {"A": 139.5, "B": 140.3}, "S4": {"A": 316.1, "B": 332.2}}
    for c in CASES:
        for R in READERS:
            assert frz[c][R]["net_mm3"] == expected[c][R]
    vox = {}
    for c in CASES:
        vox[c] = json.loads((ROOT / f"analysis/{c}/fixation_pins_2026-09-13.json").read_text())["voxel_mm"]
    per_window, frozen_below, src = {}, {}, {}
    for c in CASES:
        pth = ROOT / f"analysis/{c}/pin_artefact_corrected_v2_2026-09-14.json"
        a = json.loads(pth.read_text())
        assert a["rule"]["rule_text_sha256"] == RULE2_TEXT_SHA and a["eligible_pins"] == elig[c]
        src[c] = {"file": str(pth.relative_to(ROOT)), "sha256": sha256_file(pth)}
        per_window[c], frozen_below[c] = {}, {}
        for R in READERS:
            per_window[c][R], frozen_below[c][R] = {}, {}
            for wn, w in a["readers"][R].items():
                per_window[c][R][wn] = {"X_mm": w["X_mm"], "Y_mm": w["Y_mm"], "grid_shape": w["grid_shape"],
                                        "by_d": {}}
                frozen_below[c][R][wn] = {}
                for dk, v in w["variants"].items():
                    cnt = {p: q["footprint_cells_this_pin"] for p, q in v.get("per_pin", {}).items()}
                    per_window[c][R][wn]["by_d"][dk] = {
                        "footprint_cells_union": v["footprint"]["footprint_cells"],
                        "footprint_cells_per_pin": cnt,
                        "eligible_pins_with_cells_in_window": sorted(int(p) for p, n in cnt.items() if n > 0)}
                    frozen_below[c][R][wn][dk] = v["frozen_below_Hpre_in_footprint"]["n_cells_below_Hpre"]
    obj = {
        "_what": "",
        "declared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule_text": RULE_TEXT, "rule_text_sha256": sha256_text(RULE_TEXT),
        "parameters": PARAMS, "parameters_sha256": sha256_text(json.dumps(PARAMS, sort_keys=True)),
        "script": f"analysis/pin_artefact_rule_{TAG}.py",
        "script_sha256_at_declaration": sha256_file(Path(__file__).resolve()),
        "inherited": {
            "rule_v2_script": str(RULE2_PY.relative_to(ROOT)), "rule_v2_script_sha256": RULE2_SCRIPT_SHA,
            "rule_v2_text_sha256": RULE2_TEXT_SHA, "rule_v2_json_file_sha256": sha256_file(RULE2_JSON),
            "imported_unchanged": ["eligible_pins", "footprint"],
            "apply_v2_script_to_reuse_(gate, windows, per-window grids incl. S4 10x14)": {
                "file": str(A2_APPLY_PY.relative_to(ROOT)), "sha256_now": sha256_file(A2_APPLY_PY)},
            "measurement_path_apply_v1": {"file": str(A1_APPLY_PY.relative_to(ROOT)), "sha256": A1_APPLY_SHA},
            "qc_figure_template_v2": {"file": str(FIG2_PY.relative_to(ROOT)), "sha256_now": sha256_file(FIG2_PY)},
            "d_derivation": rule2["d_derivation_reference"]},
        "eligible_pins_per_case": elig,
        "eligibility_is_per_pin_same_for_readers_A_B": True,
        "not_eligible": {"S2": {"45": -6.5}, "S3": {"58": -11.9}, "S4": {"236": -4.56}},
        "pin_candidates_with_eligibility": rows,
        "footprint_reach_from_stored_bbox_per_reader_window": rule2["footprint_reach_from_stored_bbox_per_reader_window"],
        "eligible_pin_footprint_cells_per_case_reader_window_d_(stored_attempt2_apply_json)": per_window,
        "eligible_pin_footprint_source_files": src,
        "voxel_mm_cap_offset_per_case": vox,
        "code_level_assertions": {
            "P1_no_cell_increases": "",
            "P2_no_footprint_cell_newly_below_Hpre": "",
            "P3_outside_footprint_bit_identical": "",
            "window_level": ""},
        "deviation_from_D7d_wording": {
            "issue": "",
            "declared_form": "",
            "footprint_cells_where_literal_form_would_raise_(stored_attempt2_frozen_below_Hpre_counts)": frozen_below,
            "zhead_wording": ""},
        "frozen_values_to_reproduce_before_correction": {
            c: {R: {"net_mm3": frz[c][R]["net_mm3"], "gross_mm3": frz[c][R]["gross_mm3"]} for R in READERS}
            for c in CASES},
        "frozen_secondary_S4_10x14_to_reproduce": {"A": 360.6, "B": 379.8},
        "selftest_synthetic": st,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "volumes_loaded": False,
        "corrected_values_computed": False,
    }
    write_new(OUT, obj)
    print("declared", OUT.relative_to(ROOT), "rule_text_sha256", obj["rule_text_sha256"],
          "script_sha256", obj["script_sha256_at_declaration"], "parameters_sha256", obj["parameters_sha256"])
    print("eligible", elig)
    print("frozen-below (literal would raise)", json.dumps(frozen_below))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        print(json.dumps(selftest(), indent=1))
    elif a == ["--declare"]:
        declare()
    else:
        raise SystemExit(__doc__)

