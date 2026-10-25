#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfx_common import ROOT, SFX, CASES, READERS, write_new

VARIANT = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else None
OUTD = SFX / ("FINAL_scalefix_2026-10-06" if VARIANT is None else f"FINAL_scalefix_{VARIANT}_2026-10-06")
OUTF = "SCALEFIX_RESULTS_2026-10-06.json" if VARIANT is None else f"SCALEFIX_{VARIANT.upper()}_RESULTS_2026-10-06.json"
if "--out-dir" in sys.argv:
    OUTD = Path(sys.argv[sys.argv.index("--out-dir") + 1])
SEL = json.loads((SFX / ("PRIMARY_SELECTION_2026-10-06.json" if VARIANT is None else
                         f"PRIMARY_SELECTION_{VARIANT}_2026-10-06.json")).read_text())
RIGID_TAG = SEL.get("rigid_tag", "rigid_height")
PRIM, BOUNDS, MIR = SEL["primary_tag"], SEL["bound_tags"], SFX / SEL["mirror_new"]
COMMON = ("assessor", "pin_margin", "background", "spec_curve_step0.2")
ALL6 = [("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S2", "S3"), ("S2", "S4"), ("S3", "S4")]
NULL_BANDS = ["10-12", "12-14", "14-16", "16-18", "18-20"]


def J(p):
    return json.loads(Path(p).read_text())


def r1(x):
    return None if x is None else round(float(x), 1)


def env(members, pA):
    vals = [m["value_mm3"] for m in members]
    lo, hi = min(vals), max(vals)
    return {"min_mm3": lo, "max_mm3": hi, "n_members": len(vals),
            "argmin": [m["name"] for m in members if m["value_mm3"] == lo],
            "argmax": [m["name"] for m in members if m["value_mm3"] == hi],
            "width_mm3": r1(hi - lo), "width_pct_of_primary_A": round(100.0 * (hi - lo) / pA)}


def relation(e1, e2):
    lo, hi = max(e1["min_mm3"], e2["min_mm3"]), min(e1["max_mm3"], e2["max_mm3"])
    return {"overlap": bool(lo <= hi), "signed_gap_mm3": r1(lo - hi), "overlap_interval_mm3": [lo, hi] if lo <= hi else None}


def members_for(c, M, mem, bnds, oper):
    out = {}
    for R in READERS:
        rr = M["readers"][R]
        L = [{"family": "assessor", "name": f"primary_{R}", "value_mm3": rr["primary"]["net_mm3"]}]
        for d in ("1.0", "2.0"):
            L.append({"family": "pin_margin", "name": f"D7e_margin_d{d}_{R}", "value_mm3": rr["d7e"][f"d={d}"]["net_mm3"]})
        for k, v in rr["bg_members"].items():
            L.append({"family": "background", "name": f"{k}_{R}", "value_mm3": v["net_mm3"]})
        for k, v in mem["members"][R].items():
            L.append({"family": "background", "name": f"{k}_{R}", "value_mm3": v["net_mm3"]})
        if R == "A":
            for s in rr["specs"]:
                if s.get("admissible"):
                    L.append({"family": "spec_curve_step0.2", "name": f"spec_step0.2_frac{s['frac']}_gap{s['gap']:.0f}_A",
                              "value_mm3": s["net_step0.2_mm3"]})
                    if c == "S1":
                        for st in ("0.15", "0.3"):
                            L.append({"family": "spec_curve_raster", "name": f"spec_step{st}_frac{s['frac']}_gap{s['gap']:.0f}_A",
                                      "value_mm3": s[f"net_step{st}_mm3"]})
            if c == "S1":
                L.append({"family": "operator_scaled", "name": "operator_scaled_A", "value_mm3": oper})
        if c == "S1":
            RG = J(SFX / c / f"measure_{RIGID_TAG}.json")
            L.append({"family": "registration_sitebox", "name": f"sitebox_rigid_same_mask_{R}",
                      "value_mm3": RG["readers"][R]["primary"]["net_mm3"],
                      "note": ""})
        for nm, Mb in bnds.items():
            L.append({"family": "scale_bound", "name": f"{nm}_{R}", "value_mm3": Mb["readers"][R]["primary"]["net_mm3"]})
        out[R] = L
    return out


def views(members, pA):
    allm = members["A"] + members["B"]
    sel = lambda ms, fams: [m for m in ms if m["family"] in fams]
    exact = lambda ms: [m for m in ms if m["family"] != "scale_bound"]
    cb = COMMON + ("scale_bound",)
    v = {"common_family": {R: env(sel(members[R], COMMON), pA) for R in READERS},
         "all_exact": {R: env(exact(members[R]), pA) for R in READERS},
         "common_family_with_scale_bounds": {R: env(sel(members[R], cb), pA) for R in READERS},
         "all_exact_with_scale_bounds": {R: env(members[R], pA) for R in READERS}}
    v["common_family"]["case"] = env(sel(allm, COMMON), pA)
    v["all_exact"]["case"] = env(exact(allm), pA)
    v["common_family_with_scale_bounds"]["case"] = env(sel(allm, cb), pA)
    v["all_exact_with_scale_bounds"]["case"] = env(allm, pA)
    return v


def readings(V, prim, suffix=""):
    cf, ae = f"common_family{suffix}", f"all_exact{suffix}"
    out = {}
    for a, b in ALL6:
        g = {"common_family_case": relation(V[a][cf]["case"], V[b][cf]["case"]),
             "all_exact_case": relation(V[a][ae]["case"], V[b][ae]["case"]),
             "all_exact_A": relation(V[a][ae]["A"], V[b][ae]["A"]),
             "all_exact_B": relation(V[a][ae]["B"], V[b][ae]["B"]),
             "common_family_A": relation(V[a][cf]["A"], V[b][cf]["A"]),
             "common_family_B": relation(V[a][cf]["B"], V[b][cf]["B"])}
        rule_views = ("common_family_case", "all_exact_case", "all_exact_A", "all_exact_B")
        disj = {k: not g[k]["overlap"] for k in rule_views}
        verdict = ("separated" if all(disj.values()) else
                   "not robustly resolved" if disj["common_family_case"] else "overlapping (common family)")
        out[f"{a}_{b}"] = {"views": g, "reading": verdict,
                           "difference_A_mm3": r1(prim[b]["A"] - prim[a]["A"]), "difference_B_mm3": r1(prim[b]["B"] - prim[a]["B"]),
                           "order_A": f"{b} > {a}" if prim[b]["A"] > prim[a]["A"] else f"{a} > {b}",
                           "order_B": f"{b} > {a}" if prim[b]["B"] > prim[a]["B"] else f"{a} > {b}"}
    return out


def containment(c):
    d = J(MIR / f"analysis/{c}/null_window_check.json")
    bc = d["band_comparison"]
    above = [k for k in NULL_BANDS if k in bc and bc[k]["distinguishable_from_undisturbed_bone"]]
    contiguous = bool(above and above[0] == "10-12")
    depth = 10.0
    if contiguous:
        for k in NULL_BANDS:
            if k in above and float(k.split("-")[0]) == depth:
                depth = float(k.split("-")[1])
    cum = d["depth_cumulative"]
    below = 0.0
    prev = cum["10"]["null_corrected_cumulative_mm3"]
    for k in NULL_BANDS:
        hi = k.split("-")[1]
        if hi in cum:
            inc = cum[hi]["null_corrected_cumulative_mm3"] - prev
            prev = cum[hi]["null_corrected_cumulative_mm3"]
            if k in above:
                below += inc
    return {"n_null_windows": d["n_null_windows"], "bands_above_null": above, "contiguous_from_window_edge": contiguous,
            "contained_depth_mm": depth, "gain_below_window_mm3": round(below, 1),
            "zero_to_ten_percentile": bc["0-10"]["roi_percentile_in_null"],
            "verdict": "LOWER BOUND" if contiguous else ("contained (isolated deep band)" if above else "contained"),
            "bands": {k: {kk: v[kk] for kk in ("roi_gain_mm3", "roi_density", "null_median", "roi_percentile_in_null")}
                      for k, v in bc.items()}}


def vertical(c):
    v = J(MIR / "FINAL_ARCH_20260907/linear_vertical_gain.json")["cases"][c]
    s = v["summary_primary_noise"]
    out = {"noise_mm": v["noise"]["primary_mm"], "summary": s, "stations": []}
    for st in v["stations"]:
        out["stations"].append({k: st.get(k) for k in ("station", "crest_change_mm", "urban", "clean_for_crest_change",
                                                         "clean_for_urban", "flags", "pre_apex_class", "post_apex_class",
                                                         "aug_apex_class") if k in st})
    return out


def main():
    res = {"_what": "", "selection": SEL,
           "rules": {"declared": "analysis/scalefix_2026-10-06/PREDECLARED_RULES_2026-10-06.md",
                     "amendments": ["", "", ""]},
           "cases": {}}
    V, prim = {}, {}
    for c in CASES:
        tag = PRIM[c]
        M = J(SFX / c / f"measure_{tag}.json")
        mem = J(SFX / c / f"members_{tag}.json")
        bnds = {nm: J(SFX / c / f"measure_{bt}.json") for nm, bt in zip(("scale_low", "scale_high"), BOUNDS[c])}
        oc = J(MIR / f"analysis/{c}/outline_compare.json")
        ratio = oc["pooled_gain_ratio_operator_over_algorithm"]
        oper = round(M["readers"]["A"]["primary"]["net_mm3"] * ratio, 1)
        members = members_for(c, M, mem, bnds, oper)
        pA = M["readers"]["A"]["primary"]["net_mm3"]
        V[c] = views(members, pA)
        prim[c] = {R: M["readers"][R]["primary"]["net_mm3"] for R in READERS}
        rs = J(SFX / c / "reg_similarity.json"); us = J(SFX / c / "uncert_similarity.json")
        ra = J(SFX / c / "reg_affine.json"); ua = J(SFX / c / "uncert_affine.json")
        reg_prim = J(SFX / c / f"reg_{tag.split('_')[0]}.json") if tag.startswith("similarity") else J(SFX / c / "materialize_affine.json")
        crowns = {p: J(SFX / c / f"crowns_{p}_v3.json") for p in ("frozen", tag.split("_")[0]) if (SFX / c / f"crowns_{p}_v3.json").exists()}
        reg = {"primary_model": "similarity" if tag.startswith("similarity") else "affine",
               "similarity": {"scale": rs["final"]["scale"], "rotation_deg": rs["final"]["rotation_deg"],
                              "jackknife_se": us["jackknife_se"], "ci95": us["ci95"], "seed_sd": us["seed_sd"],
                              "perturbation": us.get("init_perturbation"), "alt_correlation_metric": us["alt_correlation_metric"],
                              "alt_graft_extent_exclusion": us["alt_graft_extent_exclusion"]["scale"],
                              "D3_unstable": us["D3_unstable"], "D3_reasons": us["D3_reasons"], "qc": rs["qc"],
                              "stages": rs["stages"]},
               "affine": {"P_diag_SAL": ra["final"]["P_diag_SAL"], "P_offdiag_SA_SL_AL": ra["final"]["P_offdiag_SA_SL_AL"],
                          "principal_stretches": ra["final"]["principal_stretches"],
                          "precondition_metric_not_worse": ra["precondition_affine_metric_not_worse"],
                          "metric_affine": ra["metric_affine"]["7"], "metric_similarity": ra["metric_similarity"]["7"],
                          "jackknife_se": ua.get("jackknife_se"), "ci95": ua.get("ci95"), "D4_pairs": ua.get("D4_pairs"),
                          "D4a_selected": ua["D4a_anisotropic_primary"]},
               "primary_qc": reg_prim["qc"], "crowns": crowns}
        case = {"primary_tag": tag, "registration": reg, "contained_depth_used_mm": M["contained_depth_mm"],
                "containment": containment(c), "vertical_gain": vertical(c),
                "operator": {"ratio": ratio, "scaled_A_mm3": oper, "stations": oc["stations"]},
                "titanium_match": M["titanium_match"], "implant_voxels": M["implant_voxels"],
                "readers": {}, "members": members, "envelopes": V[c], "members_detail": mem,
                "bounds": {nm: {"tag": bt} for nm, bt in zip(("scale_low", "scale_high"), BOUNDS[c])}}
        vx = {"S1": 0.24, "S2": 0.30, "S3": 0.16, "S4": 0.20}[c]
        case["implant_total_on_image_mm3"] = None if M["implant_voxels"] is None else round(M["implant_voxels"] * vx ** 3, 1)
        for R in READERS:
            rr = M["readers"][R]
            case["readers"][R] = {k: rr.get(k) for k in ("as_executed", "primary", "controls", "d7e", "prominence", "graft_extent",
                                                         "secondary", "extended", "threshold_sweep_gross_mm3", "bg_members")}
            case["readers"][R]["specs"] = rr["specs"]
            case["readers"][R]["bounds_primary_mm3"] = {nm: b["readers"][R]["primary"]["net_mm3"] for nm, b in bnds.items()}
        dec = {}
        TL = ("frozen_old", "frozen_height", "frozen_heightfilter", "similarity_old", "similarity_height", "similarity_heightfilter",
              "affine_height", "rigid_height", "scale1p012_height", "scalelo_height", "scalehi_height",
              "affinelo_height", "affinehi_height")
        for t in TL + (tuple(f"{t_}_{VARIANT}" for t_ in TL) if VARIANT else ()):
            p = SFX / c / f"measure_{t}.json"
            if p.exists():
                X = J(p)
                dec[t] = {R: {"background_mm": X["readers"][R]["controls"]["background_mm"],
                              "gain_before_background_mm3": X["readers"][R]["as_executed"]["gain_before_background_mm3"],
                              "as_executed_mm3": X["readers"][R]["as_executed"]["net_mm3"],
                              "primary_mm3": X["readers"][R]["primary"]["net_mm3"],
                              "aborts": X["readers"][R]["controls"]["abort_conditions"],
                              "controls": {nm: {"X_mm": w.get("X_mm"), "Y_mm": w.get("Y_mm"), "median": w.get("buccal_median_mm"),
                                                "valid": w.get("valid"), "cells": w.get("cells")}
                                           for nm, w in X["readers"][R]["controls"]["windows"].items()}}
                          for R in READERS}
        case["decomposition"] = dec
        frA = [{"name": "asexec_A", "value_mm3": M["readers"]["A"]["as_executed"]["net_mm3"]}]
        for s in M["readers"]["A"]["specs"]:
            if s.get("admissible"):
                for st in (0.15, 0.2, 0.3):
                    frA.append({"name": f"spec_asexec_step{st}_frac{s['frac']}_gap{s['gap']:.0f}", "value_mm3": s[f"asexec_step{st}_mm3"]})
        frA.append({"name": "operator_scaled_asexec_A", "value_mm3": round(M["readers"]["A"]["as_executed"]["net_mm3"] * ratio, 1)})
        frB = [{"name": "asexec_B", "value_mm3": M["readers"]["B"]["as_executed"]["net_mm3"]}]
        case["prekey_analogue_as_executed"] = {"A": env(frA, frA[0]["value_mm3"]), "B": env(frB, frA[0]["value_mm3"]),
                                               "case": env(frA + frB, frA[0]["value_mm3"]), "members": frA + frB}
        res["cases"][c] = case
    res["pairs"] = {"main": readings(V, prim), "with_scale_bounds": readings(V, prim, "_with_scale_bounds")}
    res["prekey_analogue_pairs"] = {f"{a}_{b}": relation(res["cases"][a]["prekey_analogue_as_executed"]["case"],
                                                         res["cases"][b]["prekey_analogue_as_executed"]["case"]) for a, b in ALL6}
    OUTD.mkdir(exist_ok=True)
    if VARIANT:
        res["variant"] = SEL.get("variant_note", VARIANT)
    write_new(OUTD / OUTF, res)
    for k, v in res["pairs"]["main"].items():
        print(k, v["reading"], {kk: vv["signed_gap_mm3"] for kk, vv in v["views"].items()}, v["difference_A_mm3"])
    print("--- with scale bounds")
    for k, v in res["pairs"]["with_scale_bounds"].items():
        print(k, v["reading"], {kk: vv["signed_gap_mm3"] for kk, vv in v["views"].items()})


if __name__ == "__main__":
    main()

