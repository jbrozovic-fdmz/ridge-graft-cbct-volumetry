#!/usr/bin/env python3
from __future__ import annotations

import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

ROOT = Path.cwd()
TAG = "v2_2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
AUDIT_FINAL = ROOT / "FINAL_ARCH_20260907/fixation_pins_2026-09-13.json"
OUT = ROOT / f"FINAL_ARCH_20260907/pin_artefact_rule_{TAG}.json"

PARAMS = {
    "eligibility_field": "inventory_near_site.<CASE>.pin_candidates[].median_Z_minus_Hpost_mm",
    "eligibility_threshold_mm": -3.0,
    "d_mm": 1.4,
    "d_source": "",
    "sensitivity_d_mm": [1.0, 1.4, 2.0],
    "distance_tolerance_mm": 1e-6,
    "grid_step_mm": 0.20,
    "grid_edges": "",
    "cell_centre": "",
    "windows": {"primary": "",
                "secondary_S4": ""},
    "neighbourhood": "",
    "linear_solver": "",
    "implant_term_secondary": "",
}
DECLARED_ELIGIBLE = {"S1": [], "S2": [63, 64], "S3": [70], "S4": [441, 735, 760]}
CALLER_EXPECTED = {"S1": [], "S2": [63, 64], "S3": [], "S4": [441, 760], "S4_readerB_conditional": [735]}

RULE_TEXT = ""


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p: Path, obj) -> None:
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def eligible_pins(audit_path=AUDIT_FINAL):
    inv = json.loads(Path(audit_path).read_text())["inventory_near_site"]
    thr = PARAMS["eligibility_threshold_mm"]
    out, rows = {}, {}
    for c in CASES:
        out[c], rows[c] = [], []
        for p in inv[c]["pin_candidates"]:
            m = p["median_Z_minus_Hpost_mm"]
            ok = m is not None and m >= thr
            rows[c].append({"label": p["label"], "n_voxels": p["n_voxels"], "post_grey_max": p["post_grey_max"],
                            "class": p["class"], "median_Z_minus_Hpost_mm": m, "eligible": bool(ok),
                            "in_window_footprint_A": p["in_window_footprint_A"],
                            "in_window_footprint_B": p["in_window_footprint_B"],
                            "frac_in_envelope_A": p["frac_in_envelope_A"]})
            if ok:
                out[c].append(p["label"])
        out[c].sort()
    assert out == DECLARED_ELIGIBLE, ("eligible set differs from the declared list", out)
    return out, rows


def _gap(lo, hi, c0, c1):
    return max(0.0, c0 - hi, lo - c1)


def reach_from_bbox(case_json, label, R, x_mm, y_mm, d_list):
    comp = next(c for c in case_json["components"] if c["label"] == label)
    L = comp[f"local_{R}"]
    (xa, xb), (ya, yb) = L["X_range_mm"], L["Y_range_mm"]
    s = PARAMS["grid_step_mm"]
    cx0, cx1 = x_mm[0] + s / 2, x_mm[1] - s / 2
    cy0, cy1 = y_mm[0] + s / 2, y_mm[1] - s / 2
    lb = float(np.hypot(_gap(xa, xb, cx0, cx1), _gap(ya, yb, cy0, cy1)))
    ub_cands = []
    for xe_ in (xa, xb):
        ub_cands.append(np.hypot(_gap(xe_, xe_, cx0, cx1) + s / 2,
                                 max(_gap(ya, ya, cy0, cy1), _gap(yb, yb, cy0, cy1)) + s / 2))
    for ye_ in (ya, yb):
        ub_cands.append(np.hypot(max(_gap(xa, xa, cx0, cx1), _gap(xb, xb, cx0, cx1)) + s / 2,
                                 _gap(ye_, ye_, cy0, cy1) + s / 2))
    ub = float(min(ub_cands))
    tol = PARAMS["distance_tolerance_mm"]
    return {"X_range_mm": [xa, xb], "Y_range_mm": [ya, yb], "dist_lower_bound_mm": round(lb, 3),
            "dist_upper_bound_mm": round(ub, 3),
            "reach": {f"d={d:.1f}": ("none" if lb > d + tol else "yes" if ub <= d else "possible")
                      for d in d_list}}


def footprint(xe, ye, pin_xy, d):
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    ny, nx = len(yc), len(xc)
    if pin_xy is None or not len(pin_xy):
        return np.zeros((ny, nx), bool)
    X, Y = np.meshgrid(xc, yc)
    G = np.column_stack([X.ravel(), Y.ravel()])
    dist, _ = cKDTree(np.asarray(pin_xy, float)[:, :2]).query(G, k=1)
    return (dist <= d + PARAMS["distance_tolerance_mm"]).reshape(ny, nx)


def harmonic_fill(H, F):
    H = np.asarray(H, float)
    F = np.asarray(F, bool)
    assert H.shape == F.shape and H.ndim == 2
    ny, nx = H.shape
    out = H.copy()
    info = {"footprint_cells": int(F.sum()), "dirichlet_cells": 0, "components": 0,
            "components_without_boundary": 0, "cells_set_nan_no_boundary": 0,
            "frozen_nonfinite_footprint_cells_kept_nan": int((F & ~np.isfinite(H)).sum()),
            "footprint_cells_on_window_edge": int(F[0, :].sum() + F[-1, :].sum() + F[1:-1, 0].sum()
                                                   + F[1:-1, -1].sum()) if ny > 1 and nx > 1 else int(F.sum())}
    if not F.any():
        return out, info
    known = (~F) & np.isfinite(H)
    lab, ncomp = ndi.label(F)
    info["components"] = int(ncomp)
    nbr = ((-1, 0), (1, 0), (0, -1), (0, 1))
    D = np.zeros(F.shape, bool)
    for di, dj in nbr:
        sh = np.zeros(F.shape, bool)
        src = F[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)]
        sh[max(0, di):ny - max(0, -di), max(0, dj):nx - max(0, -dj)] = src
        D |= sh & known
    info["dirichlet_cells"] = int(D.sum())
    comp_has_D = np.zeros(ncomp + 1, bool)
    for di, dj in nbr:
        for i, j in np.argwhere(D):
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and F[a, b]:
                comp_has_D[lab[a, b]] = True
    solve = F & comp_has_D[lab]
    dead = F & ~comp_has_D[lab]
    info["components_without_boundary"] = int(sum(1 for k in range(1, ncomp + 1) if not comp_has_D[k]))
    info["cells_set_nan_no_boundary"] = int(dead.sum())
    out[dead] = np.nan
    cells = np.argwhere(solve)
    if len(cells):
        idx = -np.ones(F.shape, np.int64)
        idx[tuple(cells.T)] = np.arange(len(cells))
        rows, cols, vals = [], [], []
        b = np.zeros(len(cells))
        for k, (i, j) in enumerate(cells):
            deg = 0
            for di, dj in nbr:
                a, c = i + di, j + dj
                if not (0 <= a < ny and 0 <= c < nx):
                    continue
                if solve[a, c]:
                    deg += 1
                    rows.append(k); cols.append(idx[a, c]); vals.append(-1.0)
                elif known[a, c]:
                    deg += 1
                    b[k] += H[a, c]
            rows.append(k); cols.append(k); vals.append(float(deg))
        A = sparse.csc_matrix((vals, (rows, cols)), shape=(len(cells), len(cells)))
        x = spsolve(A, b)
        keep = np.isfinite(H[tuple(cells.T)])
        out[tuple(cells[keep].T)] = x[keep]
    return out, info


def corrected_window(V, frame, Hpre, Hpost, xe, ye, background, pin_idx_SAL, implant_idx_SAL, vx, d,
                     implant_term_override=None):
    pin_idx_SAL = None if pin_idx_SAL is None or not len(pin_idx_SAL) else np.asarray(pin_idx_SAL)
    pin_xy = None if pin_idx_SAL is None else frame.to_local(pin_idx_SAL.astype(float) * vx)[:, :2]
    F = footprint(xe, ye, pin_xy, d)
    Hc, info = harmonic_fill(Hpost, F)
    m = V.measure_window(None, None, None, None, PARAMS["grid_step_mm"], background=background,
                         fields=(Hpre, Hc, xe, ye))
    if implant_term_override is not None:
        imp = implant_term_override
    else:
        imp = V.implant_volume_in_envelope(implant_idx_SAL, frame, Hpre, Hc, xe, ye, vx)["implant_in_envelope_mm3"]
    pin = (V.implant_volume_in_envelope(pin_idx_SAL, frame, Hpre, Hc, xe, ye, vx)["implant_in_envelope_mm3"]
           if pin_idx_SAL is not None else 0.0)
    return {"net_mm3": round(m["gain_mm3"] - imp - pin, 1), "gross_mm3": m["gain_mm3"],
            "implant_term_mm3": imp, "pin_term_mm3": pin,
            "net_surface_fix_only_mm3": round(m["gain_mm3"] - imp, 1),
            "coverage": m["coverage"], "area_mm2": m["area_mm2"], "fill": info,
            "_F": F, "_Hpost_corrected": Hc}


def selftest():
    res = {}
    ny, nx = 50, 50
    J, I = np.meshgrid(np.arange(nx), np.arange(ny))
    lin = 0.3 * J + 0.7 * I + 2.0
    F = (J - 25) ** 2 + (I - 25) ** 2 <= 36
    Hn, info = harmonic_fill(lin, F)
    res["interior_linear_max_abs_err"] = float(np.abs(Hn - lin).max())
    res["interior_outside_bit_identical"] = bool(np.array_equal(Hn[~F], lin[~F]))
    assert res["interior_linear_max_abs_err"] < 1e-9 and res["interior_outside_bit_identical"]
    assert info["components"] == 1 and info["components_without_boundary"] == 0
    linx = 0.5 * J + 1.0
    F2 = (J >= 20) & (J <= 30) & (I <= 5)
    H2, info2 = harmonic_fill(linx, F2)
    res["edge_neumann_max_abs_err"] = float(np.abs(H2 - linx).max())
    assert res["edge_neumann_max_abs_err"] < 1e-9 and info2["footprint_cells_on_window_edge"] > 0
    H3 = lin.copy(); H3[25, 18] = np.nan; H3[25, 25] = np.nan
    H3n, info3 = harmonic_fill(H3, F)
    res["nan_footprint_cell_kept_nan"] = bool(np.isnan(H3n[25, 25]))
    res["nan_boundary_neighbour_untouched"] = bool(np.isnan(H3n[25, 18]))
    assert res["nan_footprint_cell_kept_nan"] and res["nan_boundary_neighbour_untouched"]
    assert np.isfinite(H3n[F & np.isfinite(H3)]).all()
    H4 = np.full((10, 10), np.nan); H4[2:5, 2:5] = 1.0
    F4 = np.zeros((10, 10), bool); F4[3, 3] = True; F4[7, 7] = True
    H4n, info4 = harmonic_fill(H4, F4)
    res["no_boundary_component_nan"] = bool(np.isnan(H4n[7, 7]) and H4n[3, 3] == 1.0)
    assert res["no_boundary_component_nan"] and info4["components_without_boundary"] == 1
    bump = lin + 3.0 * F
    H5, _ = harmonic_fill(bump, F)
    res["bump_removed_max_abs_err"] = float(np.abs(H5 - lin).max())
    assert res["bump_removed_max_abs_err"] < 1e-9
    res["deterministic_repeat_array_equal"] = bool(np.array_equal(harmonic_fill(bump, F)[0], H5))
    assert res["deterministic_repeat_array_equal"]
    xe = np.arange(10.0, 20.0 + 1e-9, 0.2); ye = np.arange(15.0, 25.0 + 1e-9, 0.2)
    Fp = footprint(xe, ye, np.array([[15.1, 20.1]]), 1.0)
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    ref = np.hypot(X - 15.1, Y - 20.1) <= 1.0 + 1e-6
    res["footprint_matches_bruteforce"] = bool(np.array_equal(Fp, ref))
    res["footprint_cells_d1_single_point"] = int(Fp.sum())
    assert res["footprint_matches_bruteforce"] and Fp.shape == (50, 50)
    res["all_passed"] = True
    return res


def declare():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; not overwritten")
    v2_outputs = [p for p in glob.glob(str(ROOT / "analysis/S*/*v2_2026-09-14*"))
                  + glob.glob(str(ROOT / "FINAL_ARCH_20260907/*corrected*v2*"))]
    assert not v2_outputs, ("attempt-2 outputs already exist; declaration would not precede them", v2_outputs)
    st = selftest()
    elig, rows = eligible_pins()
    rule_v1 = json.loads((ROOT / "FINAL_ARCH_20260907/pin_artefact_rule_2026-09-14.json").read_text())
    assert rule_v1["d_mm"] == PARAMS["d_mm"] == 1.4 and rule_v1["derivation_pooled"]["k_star"] == 14
    frz = json.loads((ROOT / "FINAL_ARCH_20260907/FROZEN_RESULTS.json").read_text())["cases"]
    reach = {}
    for c in CASES:
        cj = json.loads((ROOT / f"analysis/{c}/fixation_pins_2026-09-13.json").read_text())
        reach[c] = {}
        for R in READERS:
            res = json.loads((ROOT / (f"analysis/{c}/results.json" if R == "A"
                                      else f"analysis/{c}/readerB/results.json")).read_text())
            roi = res["roi"]
            wins = {"primary_10x10": (roi["X_mm"], roi["Y_mm"])}
            if c == "S4":
                wins["secondary_10x14"] = (roi["X_mm"], [roi["Y_mm"][0], roi["Y_mm"][0] + 14.0])
            reach[c][R] = {w: {"X_mm": xy[0], "Y_mm": xy[1],
                               "pins": {str(lb): reach_from_bbox(cj, lb, R, xy[0], xy[1],
                                                                 PARAMS["sensitivity_d_mm"])
                                        for lb in elig[c]}}
                           for w, xy in wins.items()}
    mism = {c: {"data": elig[c], "caller_expected": CALLER_EXPECTED[c],
                "extra_in_data": sorted(set(elig[c]) - set(CALLER_EXPECTED[c]))}
            for c in CASES}
    obj = {
        "_what": "",
        "declared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule_text": RULE_TEXT, "rule_text_sha256": sha256_text(RULE_TEXT),
        "parameters": PARAMS,
        "parameters_sha256": sha256_text(json.dumps(PARAMS, sort_keys=True)),
        "script": "analysis/pin_artefact_rule_v2_2026-09-14.py",
        "script_sha256_at_declaration": sha256_file(Path(__file__).resolve()),
        "eligible_pins_per_case": elig,
        "eligibility_is_per_pin_same_for_readers_A_B": True,
        "pin_candidates_with_eligibility": rows,
        "footprint_reach_from_stored_bbox_per_reader_window": reach,
        "footprint_reach_note": "",
        "caller_expectation_check": mism,
        "caller_expectation_deviations": [
            "",
            ""],
        "d_derivation_reference": {"file": "FINAL_ARCH_20260907/pin_artefact_rule_2026-09-14.json",
                                   "d_mm": rule_v1["d_mm"], "k_star": rule_v1["derivation_pooled"]["k_star"],
                                   "rule_text_sha256": rule_v1["rule_text_sha256"]},
        "frozen_values_to_reproduce_before_correction": {
            c: {R: {"net_mm3": frz[c][R]["net_mm3"], "gross_mm3": frz[c][R]["gross_mm3"]} for R in READERS}
            for c in CASES},
        "frozen_secondary_S4_10x14_to_reproduce": {"A": 360.6, "B": 379.8},
        "selftest_synthetic": st,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "volumes_loaded": False,
        "corrected_values_computed": False,
    }
    for c in CASES:
        for R in READERS:
            assert obj["frozen_values_to_reproduce_before_correction"][c][R]["net_mm3"] == \
                {"S1": {"A": 105.6, "B": 104.8}, "S2": {"A": 119.9, "B": 108.0},
                 "S3": {"A": 139.5, "B": 140.3}, "S4": {"A": 316.1, "B": 332.2}}[c][R]
    write_new(OUT, obj)
    print("declared", OUT, "rule_text_sha256", obj["rule_text_sha256"],
          "script_sha256", obj["script_sha256_at_declaration"])
    print("eligible", elig)
    print(json.dumps(reach, indent=0)[:4000])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--selftest"]:
        print(json.dumps(selftest(), indent=1))
    elif a == ["--declare"]:
        declare()
    else:
        raise SystemExit(__doc__)

