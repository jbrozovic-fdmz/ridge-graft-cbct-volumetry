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
from scipy.spatial import cKDTree

ROOT = Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "analysis"))
from pipeline import volume as V

TAG = "D7e_2026-09-14"
TAG3 = "v3_2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
V3_PY = ROOT / "analysis/pin_artefact_apply_v3_2026-09-14.py"
V3_SHA = "d85e7d123cdd679c91261f574c6fce2d00ca4bb6f2c5eea5488ec7e724516a4b"
PROM_JSON = ROOT / "FINAL_ARCH_20260907/pin_prominence_D7e_2026-09-14.json"
PROM_SHA = "17fffc61f0625b260a810f06da0eb85bbdcaf9b353663cba79a37b07d06d1ad4"
CMP_JSON = ROOT / "FINAL_ARCH_20260907/pin_artefact_attempts_comparison_2026-09-14.json"
CMP_SHA = "d67b947188c1d21cd71b46cd98fd83d8f0b6e7643cf2bd3db28b1fe8ab37d2e9"
V3_FINAL = ROOT / "FINAL_ARCH_20260907/pin_artefact_corrected_v3_2026-09-14.json"
V3_FINAL_SHA = "751b61231b694fae8202463767804d5ed3520448768072b78473f502800f6073"
OUT_FINAL = ROOT / f"FINAL_ARCH_20260907/pin_artefact_primary_{TAG}.json"
TOL = 1e-6
TIE = 1e-9
QC_RADIUS_MM = 1.0


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
    assert sha256_file(V3_PY) == V3_SHA, "ABORT: attempt-3 apply script changed"
    V3 = _load("pin_artefact_apply_v3_20260914", V3_PY)
    A1, RULE1, RULE2, A2, RULE3, rule3, info3 = V3.load_all()
    assert sha256_file(PROM_JSON) == PROM_SHA, "ABORT: prominence decision table changed"
    prom = json.loads(PROM_JSON.read_text())
    assert prom["corrected_values_computed"] is False
    for c, v in prom["per_case"].items():
        if v.get("file"):
            assert sha256_file(ROOT / v["file"]) == v["sha256"], ("per-case prominence file changed", c)
    dec = {}
    for t in prom["table"]:
        assert t["cap_applied"] == (t["decision"] == "prominent")
        if t["decision"] == "prominent":
            assert t["P_mm"] is not None and t["P_mm"] > t["voxel_mm"]
        elif t["decision"] == "not prominent":
            assert t["P_mm"] is not None and t["P_mm"] <= t["voxel_mm"]
        else:
            assert t["decision"] == "undetermined" and t["P_mm"] is None
        if t["QC_10x14_decision"] is not None:
            assert t["QC_10x14_decision"] == t["decision"], ("10x14 QC decision disagrees", t)
        assert abs(t["voxel_mm"] - rule3["voxel_mm_cap_offset_per_case"][t["case"]]) < 1e-12
        dec.setdefault(t["case"], {}).setdefault(t["reader"], {})[int(t["pin"])] = t
    for c in CASES:
        el = rule3["eligible_pins_per_case"][c]
        for R in READERS:
            assert sorted(dec.get(c, {}).get(R, {}).keys()) == sorted(el), (c, R)
    info = {"attempt3_apply_script": str(V3_PY.relative_to(ROOT)), "attempt3_apply_sha256": V3_SHA,
            "attempt3_rule": info3,
            "prominence_table": str(PROM_JSON.relative_to(ROOT)), "prominence_table_sha256": PROM_SHA,
            "prominence_table_utc": prom["utc"],
            "eligible_pins_per_case": rule3["eligible_pins_per_case"],
            "d_mm": 1.4, "sensitivity_d_mm": [1.0, 1.4, 2.0],
            "voxel_mm_per_case": rule3["voxel_mm_cap_offset_per_case"]}
    return A1, RULE1, RULE2, A2, RULE3, rule3, V3, dec, info


def gate(case):
    A1, RULE1, RULE2, A2, RULE3, rule3, V3, dec, info = load_all()
    p = ROOT / f"analysis/{case}/pin_artefact_gate_{TAG}.json"
    if p.exists():
        raise SystemExit(f"{p} exists; not overwritten")
    for cc in CASES:
        assert not (ROOT / f"analysis/{cc}/pin_artefact_corrected_{TAG}.json").exists(), \
            "ABORT: a D7e corrected value already exists; the gate must precede every corrected value"
    assert not OUT_FINAL.exists()
    t0 = time.time()
    c = A1.setup(case, RULE1)
    Q = A1.post_clouds(c, c.post)
    out = {"case": case, "_what": "",
           "rule": info, "prominence_table_present_before_gate": True, "pins": c.pin_rows,
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
    print(f"[{case}] D7e gate written; peak RSS {out['peak_rss_GiB']} GiB", flush=True)


def dist_field(xe, ye, pin_xy):
    xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    G = np.column_stack([X.ravel(), Y.ravel()])
    d, _ = cKDTree(np.asarray(pin_xy, float)[:, :2]).query(G, k=1)
    return d.reshape(X.shape)


def residual_above_titanium(H, near, zt):
    h = H[near]
    h = h[np.isfinite(h)]
    return {"volume_mm3": round(float(np.clip(h - zt, 0, None).sum() * 0.04), 3),
            "cells_above": int((h > zt).sum()),
            "peak_H_minus_titanium_max_Z_mm": round(float(h.max() - zt), 2) if len(h) else None}


def d7e_variant(A1, RULE2, RULE3, c, R, w, elig, pin_vox, pin_idx, loc, prom_flag, d, imp_term, v3_case, npz3, wn):
    frame = c.frames[R]; vx = c.vx
    Hpre, Hpost, xe, ye, bg = w["Hpre"], w["Hpost"], w["xe"], w["ye"], w["background"]
    b_pre, b_post = Hpre.tobytes(), Hpost.tobytes()
    step = 0.2
    m0 = V.measure_window(None, None, None, None, step, background=bg, fields=(Hpre, Hpost, xe, ye))
    assert round(m0["gain_mm3"] - imp_term, 1) == w["frozen_net"], "frozen identity failed"
    dk = f"d={d:.1f}"
    k3 = f"{R}_{wn}_d{d:.1f}"
    o3 = RULE3.corrected_window(V, frame, Hpre, Hpost, xe, ye, bg, pin_idx, vx, d, imp_term, w["frozen_net"])
    F_all, Z, Hc3 = o3["_F"], o3["_Zhead"], o3["_Hpost_corrected"]
    assert o3["net_mm3"] == v3_case["readers"][R][wn]["variants"][dk]["net_mm3"], "attempt-3 net not reproduced"
    assert np.array_equal(F_all, npz3[k3 + "_F"]) and Hc3.tobytes() == npz3[k3 + "_Hc"].tobytes(), \
        "attempt-3 arrays not reproduced"
    assert Hpre.tobytes() == npz3[f"{R}_{wn}_Hpre"].tobytes() and Hpost.tobytes() == npz3[f"{R}_{wn}_Hpost"].tobytes()
    ny, nx = Hpost.shape
    ties_conflict = np.zeros((ny, nx), bool)
    nearest = np.full((ny, nx), -1, np.int64)
    if elig:
        Dst = np.stack([dist_field(xe, ye, loc[L][:, :2]) for L in elig])
        assert np.array_equal(Dst.min(0) <= d + TOL, F_all), "union of per-pin distances != attempt-3 footprint"
        order = np.argsort(Dst, axis=0, kind="stable")
        i0 = order[0]
        nearest = np.array(elig)[i0]
        pflag = np.array([prom_flag[L] for L in elig])
        cap_nearest = pflag[i0]
        if len(elig) > 1:
            d0 = np.take_along_axis(Dst, order[:1], 0)[0]
            i1 = order[1]
            d1 = np.take_along_axis(Dst, order[1:2], 0)[0]
            ties_conflict = F_all & (np.abs(d1 - d0) <= TIE) & (pflag[i0] != pflag[i1])
        Fcap = F_all & cap_nearest & ~ties_conflict
    else:
        Fcap = np.zeros((ny, nx), bool)
    Hc = Hpost.copy()
    Hc[Fcap] = Hc3[Fcap]
    RULE3.assert_cell_properties(Hpre, Hpost, Hc, Fcap)
    RULE3.assert_cell_properties(Hpre, Hpost, Hc, F_all)
    fin = np.isfinite(Hpost)
    assert np.array_equal(np.isfinite(Hc), fin) and np.all(Hc[fin] <= Hpost[fin]), "P1"
    with np.errstate(invalid="ignore"):
        newly = Fcap & np.isfinite(Hpre) & fin & (Hc < Hpre) & ~(Hpost < Hpre)
    assert not newly.any(), "P2"
    assert Hc[~Fcap].tobytes() == Hpost[~Fcap].tobytes(), "P3"
    assert Hpre.tobytes() == b_pre and Hpost.tobytes() == b_post, "frozen inputs modified"
    m = V.measure_window(None, None, None, None, step, background=bg, fields=(Hpre, Hc, xe, ye))
    pin_t = (V.implant_volume_in_envelope(pin_idx, frame, Hpre, Hc, xe, ye, vx)["implant_in_envelope_mm3"]
             if pin_idx is not None else 0.0)
    net = round(m["gain_mm3"] - imp_term - pin_t, 1)
    assert m["coverage"] == m0["coverage"] and m["cells"] == m0["cells"], "coverage changed"
    assert m["gain_mm3"] <= m0["gain_mm3"], "gross increased"
    assert net <= w["frozen_net"], "net increased"
    g0 = A1.gain_unrounded(Hpre, Hpost, bg)
    g1 = A1.gain_unrounded(Hpre, Hc, bg)
    pin_u = A1.envelope_overlap(pin_idx, frame, Hpre, Hc, xe, ye, vx) if pin_idx is not None else 0.0
    pin_frozen_u = A1.envelope_overlap(pin_idx, frame, Hpre, Hpost, xe, ye, vx) if pin_idx is not None else 0.0
    assert pin_t == round(pin_u, 1)
    metal_only_net = round(m0["gain_mm3"] - imp_term - round(pin_frozen_u, 1), 1) if pin_idx is not None else w["frozen_net"]
    n_prom = sum(prom_flag.values()) if elig else 0
    if elig and n_prom == len(elig):
        assert Hc.tobytes() == Hc3.tobytes() and net == o3["net_mm3"], "all pins prominent but D7e != attempt 3"
    if not elig or n_prom == 0:
        assert Hc.tobytes() == Hpost.tobytes(), "no prominent pin but surface changed"
    lowered = fin & (Hc < Hpost)
    rec = {"d_mm": d, "net_mm3": net, "frozen_net_mm3": w["frozen_net"],
           "delta_net_mm3": round(net - w["frozen_net"], 1),
           "delta_pct": round(100.0 * (net - w["frozen_net"]) / w["frozen_net"], 1),
           "gross_mm3": m["gain_mm3"], "frozen_gross_mm3": m0["gain_mm3"], "implant_term_frozen_mm3": imp_term,
           "metal_term_all_eligible_pins_mm3": pin_t, "net_cap_only_mm3": round(m["gain_mm3"] - imp_term, 1),
           "coverage": m["coverage"], "coverage_frozen": w["frozen_coverage"], "area_mm2": m["area_mm2"],
           "decomposition_unrounded": {"cap_term_mm3": r(g1 - g0, 3), "metal_term_mm3": r(-pin_u, 3),
                                       "delta_net_mm3": r((g1 - g0) - pin_u, 3),
                                       "metal_on_FROZEN_fields_mm3_(reference)": r(pin_frozen_u, 3)},
           "decomposition_rounded": {"cap_term_mm3": round(round(m["gain_mm3"] - imp_term, 1) - w["frozen_net"], 1),
                                     "metal_term_mm3": round(-pin_t, 1)},
           "REFERENCE_metal_only_eligible_pins_on_frozen_fields_net_mm3": metal_only_net,
           "REFERENCE_attempt3_net_mm3_reproduced": o3["net_mm3"],
           "footprint_cells_all_eligible": int(F_all.sum()),
           "capped_cells_Fcap": int(Fcap.sum()),
           "cells_lowered": int(lowered.sum()),
           "max_lowering_mm": r(float((Hpost - Hc)[lowered].max()), 3) if lowered.any() else 0.0,
           "cells_floored_at_Hpre": int((Fcap & fin & np.isfinite(Hpre) & (Hc == Hpre) & (Hpost > Hpre)).sum()),
           "footprint_cells_not_capped_(nearest_pin_not_prominent)": int((F_all & ~Fcap).sum()),
           "tie_conflict_cells_not_capped": int(ties_conflict.sum()),
           "cells_newly_below_Hpre_asserted_zero": int(newly.sum()),
           "attempt3_arrays_reproduced_bytes": True,
           "properties_P1_P2_P3_asserted_on_Fcap_and_F_all": True,
           "per_pin": {}}
    if elig and 0 < n_prom < len(elig) and Fcap.any():
        Lp = np.vstack([loc[L] for L in elig if prom_flag[L]])
        Fp, Zp = RULE3.zhead_field(xe, ye, Lp, d)
        assert np.all(Fp[Fcap])
        capq = Fcap & fin
        Hq = Hpost.copy()
        m1 = capq & np.isfinite(Hpre); m2 = capq & ~np.isfinite(Hpre)
        Hq[m1] = np.minimum(Hpost[m1], np.maximum(Hpre[m1], Zp[m1] + vx))
        Hq[m2] = np.minimum(Hpost[m2], Zp[m2] + vx)
        RULE3.assert_cell_properties(Hpre, Hpost, Hq, Fcap)
        mq = V.measure_window(None, None, None, None, step, background=bg, fields=(Hpre, Hq, xe, ye))
        pq = V.implant_volume_in_envelope(pin_idx, frame, Hpre, Hq, xe, ye, vx)["implant_in_envelope_mm3"]
        rec["QC_Zhead_from_prominent_pins_only_NOT_RULE"] = {
            "capped_cells_where_Zhead_differs": int((Fcap & (Zp != Z)).sum()),
            "net_mm3": round(mq["gain_mm3"] - imp_term - pq, 1)}
    for L in elig:
        near = RULE2.footprint(xe, ye, loc[L][:, :2], QC_RADIUS_MM)
        Fp = RULE2.footprint(xe, ye, loc[L][:, :2], d)
        zt = float(loc[L][:, 2].max())
        rec["per_pin"][str(L)] = {
            "prominent_cap_applied": bool(prom_flag[L]),
            "footprint_cells_this_pin": int(Fp.sum()),
            "footprint_cells_nearest_to_this_pin": int((F_all & (nearest == L)).sum()),
            "capped_cells_nearest_to_this_pin": int((Fcap & (nearest == L)).sum()),
            "cells_lowered_nearest_to_this_pin": int((lowered & (nearest == L)).sum()),
            "this_pin_footprint_cells_capped_(any_nearest)": int((Fp & Fcap).sum()),
            "titanium_max_Z_mm": r(zt, 4), "cells_within_1.0mm": int(near.sum()),
            "residual_above_titanium_1mm_FROZEN": residual_above_titanium(Hpost, near, zt) if near.any() else None,
            "residual_above_titanium_1mm_D7e": residual_above_titanium(Hc, near, zt) if near.any() else None}
    return rec, F_all, Fcap, Z, Hc


def apply(case):
    A1, RULE1, RULE2, A2, RULE3, rule3, V3, dec, info = load_all()
    gates = {}
    for cc in CASES:
        gp = ROOT / f"analysis/{cc}/pin_artefact_gate_{TAG}.json"
        assert gp.exists(), f"ABORT: D7e gate file missing for {cc}; run --gate for all four cases first"
        g = json.loads(gp.read_text())
        assert g["all_assertions_passed"] is True and g["rule"]["prominence_table_sha256"] == PROM_SHA
        for R in READERS:
            assert g["readers"][R]["net_mm3"] == A1.EXPECTED_NET[cc][R]
            if cc in A1.EXPECTED_SECONDARY:
                assert g["readers"][R]["secondary"]["net_mm3"] == A1.EXPECTED_SECONDARY[cc][R]
        gates[cc] = {R: g["readers"][R]["net_mm3"] for R in READERS}
        if cc in A1.EXPECTED_SECONDARY:
            gates[cc]["secondary_10x14"] = {R: g["readers"][R]["secondary"]["net_mm3"] for R in READERS}
        gates[cc]["gate_utc"] = g["utc"]
    assert PROM_JSON.stat().st_mtime < min((ROOT / f"analysis/{cc}/pin_artefact_gate_{TAG}.json").stat().st_mtime
                                           for cc in CASES), "prominence table not older than the gates"
    p = ROOT / f"analysis/{case}/pin_artefact_corrected_{TAG}.json"
    pz = ROOT / f"analysis/{case}/pin_artefact_fields_{TAG}.npz"
    for q in (p, pz):
        if q.exists():
            raise SystemExit(f"{q} exists; not overwritten")
    v3_case = json.loads((ROOT / f"analysis/{case}/pin_artefact_corrected_{TAG3}.json").read_text())
    assert sha256_file(ROOT / v3_case["fields_npz"]) == v3_case["fields_npz_sha256"]
    npz3 = np.load(ROOT / v3_case["fields_npz"])
    t0 = time.time()
    c = A1.setup(case, RULE1)
    assert c.vx == rule3["voxel_mm_cap_offset_per_case"][case]
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
    assert set(got_rows) == set(decl_rows)
    for lb in elig:
        assert decl_rows[lb]["eligible"] and got_rows[lb]["n_voxels"] == decl_rows[lb]["n_voxels"]
    pin_vox = {lb: np.argwhere(c.lbl == lb) + c.lo for lb in elig}
    pin_idx = np.vstack([pin_vox[lb] for lb in elig]) if elig else None
    out = {"case": case, "_what": "", "rule": info,
           "gate_frozen_values_reproduced_D7e": gates, "eligible_pins": elig,
           "eligible_pin_voxels": {str(lb): int(len(v)) for lb, v in pin_vox.items()},
           "voxel_mm": c.vx, "mem_available_GiB_at_load": c.mem_available_GiB_at_load,
           "prominence_decisions_used": {R: {str(L): {k: dec[case][R][L][k] for k in
                                                    ("P_mm", "decision", "cap_applied", "core_valid_cells",
                                                     "ring_valid_cells")} for L in elig} for R in READERS}
           if elig else {}, "readers": {}}
    npz = {}
    for R in READERS:
        frame = c.frames[R]
        loc = {L: frame.to_local(pin_vox[L].astype(float) * c.vx) for L in elig}
        prom_flag = {L: bool(dec[case][R][L]["cap_applied"]) for L in elig}
        for L in elig:
            pz_ = json.loads((ROOT / f"analysis/{case}/pin_prominence_{TAG}.json").read_text())["readers"][R]["pins"][str(L)]
            assert abs(round(float(loc[L][:, 2].max()), 4) - pz_["Ztop_mm"]) < 1e-9 and len(pin_vox[L]) == pz_["titanium_voxels"], \
                ("pin identity differs from the prominence step", R, L)
        wins = A2.windows_for(c, R, base[R], A1)
        out["readers"][R] = {}
        for wn, w in wins.items():
            imp_term = (base[R]["implant_in_envelope_mm3"] if w["implant_override"] is None
                        else float(w["implant_override"]))
            wrec = {"grid_shape": list(w["Hpre"].shape), "frozen_net_mm3": w["frozen_net"],
                    "implant_term_frozen_mm3": imp_term, "background_frozen_mm": r(w["background"], 4),
                    "prominence_decision_source": "",
                    "variants": {}}
            key0 = f"{R}_{wn}"
            npz[f"{key0}_Hpre"] = w["Hpre"]; npz[f"{key0}_Hpost"] = w["Hpost"]
            npz[f"{key0}_xe"] = w["xe"]; npz[f"{key0}_ye"] = w["ye"]
            for d in info["sensitivity_d_mm"]:
                rec, F_all, Fcap, Z, Hc = d7e_variant(A1, RULE2, RULE3, c, R, w, elig, pin_vox, pin_idx, loc,
                                                     prom_flag, d, imp_term, v3_case, npz3, wn)
                rec["role"] = "PRIMARY" if d == info["d_mm"] else "sensitivity"
                wrec["variants"][f"d={d:.1f}"] = rec
                npz[f"{key0}_d{d:.1f}_Fall"] = F_all; npz[f"{key0}_d{d:.1f}_Fcap"] = Fcap
                npz[f"{key0}_d{d:.1f}_Zhead"] = Z; npz[f"{key0}_d{d:.1f}_Hc"] = Hc
                print(f"  {case}/{R} {wn} d={d:.1f}: D7e net {rec['net_mm3']} (frozen {w['frozen_net']}; cap "
                      f"{rec['decomposition_unrounded']['cap_term_mm3']}, metal {rec['decomposition_unrounded']['metal_term_mm3']}; "
                      f"Fall {rec['footprint_cells_all_eligible']} Fcap {rec['capped_cells_Fcap']} lowered {rec['cells_lowered']}; "
                      f"attempt3 {rec['REFERENCE_attempt3_net_mm3_reproduced']}; metal-only {rec['REFERENCE_metal_only_eligible_pins_on_frozen_fields_net_mm3']}; "
                      f"ties {rec['tie_conflict_cells_not_capped']})", flush=True)
            out["readers"][R][wn] = wrec
    out["peak_rss_GiB"] = r(A1.rss_gib(), 2)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(pz, "xb") as fh:
        np.savez_compressed(fh, **npz)
    out["fields_npz"] = str(pz.relative_to(ROOT))
    out["fields_npz_sha256"] = sha256_file(pz)
    out["script"] = f"analysis/pin_artefact_apply_{TAG}.py"
    out["script_sha256"] = sha256_file(Path(__file__).resolve())
    write_new(p, out)
    print(f"[{case}] written {p.relative_to(ROOT)} ({out['runtime_s']} s, peak RSS {out['peak_rss_GiB']} GiB)", flush=True)


def collect():
    A1, RULE1, RULE2, A2, RULE3, rule3, V3, dec, info = load_all()
    assert sha256_file(CMP_JSON) == CMP_SHA and sha256_file(V3_FINAL) == V3_FINAL_SHA
    cmpj = {(t["case"], t["reader"], t["window"]): t for t in json.loads(CMP_JSON.read_text())["table"]}
    this_sha = sha256_file(Path(__file__).resolve())
    per, table = {}, []
    for c in CASES:
        a = json.loads((ROOT / f"analysis/{c}/pin_artefact_corrected_{TAG}.json").read_text())
        g = json.loads((ROOT / f"analysis/{c}/pin_artefact_gate_{TAG}.json").read_text())
        assert g["all_assertions_passed"] and sha256_file(ROOT / a["fields_npz"]) == a["fields_npz_sha256"]
        assert a["script_sha256"] == this_sha, ("apply output produced by a different script version", c)
        assert g["utc"] < a["utc"]
        per[c] = a
        for R in READERS:
            for wn, w in a["readers"][R].items():
                v = w["variants"]; p = v["d=1.4"]; cm = cmpj[(c, R, wn)]
                assert cm["frozen_net_mm3"] == w["frozen_net_mm3"]
                assert cm["attempt3_pin_head_cap_d1.4_mm3"] == p["REFERENCE_attempt3_net_mm3_reproduced"]
                assert cm["attempt3_sensitivity_mm3"] == {k: vv["REFERENCE_attempt3_net_mm3_reproduced"] for k, vv in v.items()}
                table.append({
                    "case": c, "reader": R, "window": wn, "eligible_pins": a["eligible_pins"],
                    "prominent_pins_capped": [int(L) for L, q in (a["prominence_decisions_used"].get(R) or {}).items()
                                              if q["cap_applied"]],
                    "frozen_net_mm3": w["frozen_net_mm3"],
                    "D7e_PRIMARY_net_mm3_d1.4": p["net_mm3"], "delta_mm3": p["delta_net_mm3"], "delta_pct": p["delta_pct"],
                    "D7e_sensitivity_net_mm3": {k: vv["net_mm3"] for k, vv in v.items()},
                    "cap_term_mm3_unrounded": p["decomposition_unrounded"]["cap_term_mm3"],
                    "metal_term_mm3_unrounded": p["decomposition_unrounded"]["metal_term_mm3"],
                    "capped_cells_Fcap": p["capped_cells_Fcap"], "cells_lowered": p["cells_lowered"],
                    "footprint_cells_all_eligible": p["footprint_cells_all_eligible"],
                    "tie_conflict_cells": p["tie_conflict_cells_not_capped"],
                    "per_d": {k: {"net_mm3": vv["net_mm3"], "cap_term_mm3": vv["decomposition_unrounded"]["cap_term_mm3"],
                                  "metal_term_mm3": vv["decomposition_unrounded"]["metal_term_mm3"],
                                  "capped_cells_Fcap": vv["capped_cells_Fcap"], "cells_lowered": vv["cells_lowered"],
                                  "QC_Zhead_prominent_only_net_mm3": (vv.get("QC_Zhead_from_prominent_pins_only_NOT_RULE") or {}).get("net_mm3"),
                                  "QC_Zhead_prominent_only_cells_differing": (vv.get("QC_Zhead_from_prominent_pins_only_NOT_RULE") or {}).get("capped_cells_where_Zhead_differs")}
                              for k, vv in v.items()},
                    "per_pin_d1.4": {L: {kk: q[kk] for kk in ("prominent_cap_applied", "capped_cells_nearest_to_this_pin",
                                                              "footprint_cells_nearest_to_this_pin")}
                                     | {"residual_1mm_frozen_mm3": (q["residual_above_titanium_1mm_FROZEN"] or {}).get("volume_mm3"),
                                        "residual_1mm_D7e_mm3": (q["residual_above_titanium_1mm_D7e"] or {}).get("volume_mm3")}
                                     for L, q in p["per_pin"].items()},
                    "side_by_side_d1.4_mm3": {
                        "frozen": w["frozen_net_mm3"],
                        "metal_only_2026-09-13_(all_pin_candidates)": cm["metal_only_2026-09-13_mm3"],
                        "metal_only_eligible_pins_recomputed": p["REFERENCE_metal_only_eligible_pins_on_frozen_fields_net_mm3"],
                        "attempt1_voxel_inpaint": cm["attempt1_voxel_inpaint_d1.4_mm3"],
                        "attempt2_harmonic_fill": cm["attempt2_harmonic_fill_d1.4_mm3"],
                        "attempt3_pin_head_cap": cm["attempt3_pin_head_cap_d1.4_mm3"],
                        "D7e_prominent_pin_cap": p["net_mm3"]},
                    "sensitivity_ranges_mm3": {"attempt1_d1.1/1.4/1.7": cm["attempt1_sensitivity_mm3"],
                                               "attempt2_d1.0/1.4/2.0": cm["attempt2_sensitivity_mm3"],
                                               "attempt3_d1.0/1.4/2.0": cm["attempt3_sensitivity_mm3"],
                                               "D7e_d1.0/1.4/2.0": {k: vv["net_mm3"] for k, vv in v.items()}}})
    fin = {"_what": "",
           "date": "", "rule": info,
           "order_of_operations": {
               "prominence_table_utc": info["prominence_table_utc"],
               "gates_utc": {c: json.loads((ROOT / f"analysis/{c}/pin_artefact_gate_{TAG}.json").read_text())["utc"] for c in CASES},
               "apply_utc": {c: per[c]["utc"] for c in CASES}},
           "gate": "",
           "properties_asserted_every_variant": ["",
                                                 "", "",
                                                 "",
                                                 ""],
           "columns": {"metal_only_2026-09-13_(all_pin_candidates)": "",
                       "metal_only_eligible_pins_recomputed": "",
                       "attempt1/2/3": ""},
           "sources": {"comparison": {"file": str(CMP_JSON.relative_to(ROOT)), "sha256": CMP_SHA},
                       "attempt3_final": {"file": str(V3_FINAL.relative_to(ROOT)), "sha256": V3_FINAL_SHA},
                       "prominence": {"file": str(PROM_JSON.relative_to(ROOT)), "sha256": PROM_SHA}},
           "table": table,
           "peak_rss_GiB_apply": {c: per[c]["peak_rss_GiB"] for c in CASES},
           "peak_rss_GiB_gate": {c: json.loads((ROOT / f"analysis/{c}/pin_artefact_gate_{TAG}.json").read_text())["peak_rss_GiB"] for c in CASES},
           "per_case_files": [f"analysis/{c}/pin_artefact_corrected_{TAG}.json" for c in CASES],
           "script": f"analysis/pin_artefact_apply_{TAG}.py", "script_sha256": this_sha,
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(OUT_FINAL, fin)
    for t in table:
        s = t["side_by_side_d1.4_mm3"]
        print(t["case"], t["reader"], t["window"], "capped", t["prominent_pins_capped"], "| frozen", s["frozen"],
              "metal-only", s["metal_only_2026-09-13_(all_pin_candidates)"], s["metal_only_eligible_pins_recomputed"],
              "a1", s["attempt1_voxel_inpaint"], "a2", s["attempt2_harmonic_fill"], "a3", s["attempt3_pin_head_cap"],
              "D7e", s["D7e_prominent_pin_cap"], t["D7e_sensitivity_net_mm3"], "cap", t["cap_term_mm3_unrounded"],
              "metal", t["metal_term_mm3_unrounded"], "Fcap", t["capped_cells_Fcap"], "lowered", t["cells_lowered"],
              "ties", t["tie_conflict_cells"])


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

