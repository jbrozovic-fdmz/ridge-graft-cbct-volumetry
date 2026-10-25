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
from sfx_common import geom, V, READERS, post_path, sha256_file, write_new, rss_gib

OUT = Path(__file__).resolve().parent
TAG = "2026-10-24"
GRID = 0.2
PAPER = {"S1": "S1", "S3": "X1", "S2": "S2", "S4": "X2"}
ANALYSES = {"original": {"S1": "frozen_old_final2", "S2": "frozen_old_final2", "S3": "frozen_old_final2", "S4": "frozen_old_final2"},
            "revised": {"S1": "similarity_height_metalctl", "S2": "affine_height_metalctl", "S3": "similarity_height",
                        "S4": "similarity_height_metalctl"}}
L0 = 450.0
NPTS = 400000
T = np.round(np.arange(-3.0, 8.0001, 0.025), 4)


def J(p):
    return json.loads(Path(p).read_text())


def volumes():
    outp = OUT / f"BGOFFSET_VOLUMES_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    res, inputs = {}, {}
    for an, tags in ANALYSES.items():
        res[an] = {}
        for c in ("S1", "S3", "S2", "S4"):
            mj = SFXD / c / f"measure_{tags[c]}.json"
            M = J(mj); Z = np.load(ROOT / M["fields_npz"])
            inputs[str(mj.relative_to(ROOT))] = sha256_file(mj); inputs[M["fields_npz"]] = sha256_file(ROOT / M["fields_npz"])
            res[an][PAPER[c]] = {"file_code": c, "tag": tags[c], "readers": {}}
            for R in READERS:
                rr = M["readers"][R]
                Hpre, Hpost, Hc = Z[f"{R}_primary_10x10_Hpre"], Z[f"{R}_primary_10x10_Hpost"], Z[f"{R}_primary_10x10_d1.4_Hc"]
                xe, ye = Z[f"{R}_primary_10x10_xe"], Z[f"{R}_primary_10x10_ye"]
                bg = rr["controls"]["background_unrounded"]
                imp, pin = rr["as_executed"]["implant_in_envelope_mm3"], rr["primary"]["pin_metal_mm3"]
                mw = lambda H, b: V.measure_window(None, None, None, None, GRID, background=b, fields=(Hpre, H, xe, ye))
                e_bg, e_0 = mw(Hpost, bg), mw(Hpost, 0.0)
                p_bg, p_0 = mw(Hc, bg), mw(Hc, 0.0)
                rec = {"background_mm": rr["controls"]["background_mm"], "background_unrounded": bg,
                       "as_executed_net_mm3": round(e_bg["gain_mm3"] - imp, 1),
                       "as_executed_net_without_offset_mm3": round(e_0["gain_mm3"] - imp, 1),
                       "pin_corrected_net_mm3": round(p_bg["gain_mm3"] - imp - pin, 1),
                       "pin_corrected_net_without_offset_mm3": round(p_0["gain_mm3"] - imp - pin, 1),
                       "implant_mm3": imp, "pin_metal_mm3": pin, "measured_area_mm2": p_bg["area_mm2"],
                       "offset_nominal_mm3": round(bg * p_bg["area_mm2"], 1),
                       "offset_effective_mm3_as_executed": round(e_0["gain_mm3"] - e_bg["gain_mm3"], 1),
                       "offset_effective_mm3_pin_corrected": round(p_0["gain_mm3"] - p_bg["gain_mm3"], 1),
                       "combined_split": rr["controls"]["registration_accuracy"],
                       "windows": {nm: {k: w.get(k) for k in ("valid", "X_mm", "Y_mm", "buccal_median_mm", "lingual_outward_mm",
                                                              "rigid_shift_mm", "dilation_mm", "excluded_by_config",
                                                              "excluded_by_metal_rule")}
                                   for nm, w in rr["controls"]["windows"].items()}}
                gate = {"as_executed": [rec["as_executed_net_mm3"], rr["as_executed"]["net_mm3"]],
                        "pin_corrected": [rec["pin_corrected_net_mm3"], rr["primary"]["net_mm3"]],
                        "gain_before_background": [e_0["gain_mm3"], rr["as_executed"]["gain_before_background_mm3"]]}
                rec["gate"] = {k: {"recomputed": v[0], "stored": v[1], "equal": v[0] == v[1]} for k, v in gate.items()}
                res[an][PAPER[c]]["readers"][R] = rec
    gate_ok = all(g["equal"] for an in res.values() for cc in an.values() for r in cc["readers"].values() for g in r["gate"].values())
    pairs = {}
    for an in res:
        pairs[an] = {}
        for R in READERS:
            for basis in ("as_executed", "pin_corrected"):
                for wo in ("", "_without_offset"):
                    k = f"{basis}_net{wo}_mm3"
                    v = {cc: res[an][cc]["readers"][R][k] for cc in res[an]}
                    pairs[an][f"{R}_{basis}{wo or '_with_offset'}"] = {
                        "X1_minus_S1_mm3": round(v["X1"] - v["S1"], 1), "X2_minus_S2_mm3": round(v["X2"] - v["S2"], 1),
                        "implant_pair_order": "X1 > S1" if v["X1"] > v["S1"] else "S1 > X1",
                        "no_implant_pair_order": "X2 > S2" if v["X2"] > v["S2"] else "S2 > X2", "values": v}
    variants = {}
    A = J(SFXD / "FINAL_scalefix_metalctl_2026-10-06/SCALEFIX_METALCTL_RESULTS_2026-10-06.json")
    for c in ("S1", "S3", "S2", "S4"):
        variants[PAPER[c]] = {}
        for t, v in A["cases"][c]["decomposition"].items():
            mj = SFXD / c / f"measure_{t}.json"
            M = J(mj) if mj.exists() else None
            variants[PAPER[c]][t] = {R: {"background_mm": v[R]["background_mm"],
                                         "split": (M["readers"][R]["controls"]["registration_accuracy"] if M else None),
                                         "windows_buccal_median_mm": {nm: w.get("median") for nm, w in v[R]["controls"].items()},
                                         "as_executed_mm3": v[R]["as_executed_mm3"], "primary_mm3": v[R]["primary_mm3"],
                                         "gain_before_background_mm3": v[R]["gain_before_background_mm3"]} for R in READERS}
    S1reg = A["cases"]["S1"]["registration"]["similarity"]
    out = {"_what": "",
           "definitions": {"original": "",
                           "revised": "primary analysis",
                           "offset_nominal_mm3": "",
                           "offset_effective_mm3": ""},
           "gate_passed": gate_ok, "cases": res, "pairs": pairs, "offset_by_registration_variant": variants,
           "S1_similarity_fit": {k: S1reg.get(k) for k in ("scale", "ci95", "jackknife_se", "D3_unstable", "D3_reasons",
                                                            "alt_correlation_metric", "alt_graft_extent_exclusion")},
           "inputs_sha256": inputs, "script": str(Path(__file__).resolve().relative_to(ROOT)),
           "script_sha256": sha256_file(Path(__file__).resolve()), "peak_rss_GiB": round(rss_gib(), 2),
           "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    print("gate", gate_ok)
    for an in pairs:
        for k, v in pairs[an].items():
            print(an, k, v["values"], v["implant_pair_order"], v["X1_minus_S1_mm3"])
    if not gate_ok:
        raise SystemExit("gate failed")


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
    return {"cA": float(cA), "cL": float(cL), "bins": 0.5 * (bins[1:] + bins[:-1]), "R": Rb}


def maxilla_keep(fr, P, am, y_min=7.0, lateral_max=12.0):
    loc = fr.to_local(P)
    ang = np.degrees(np.arctan2(P[:, 2] - am["cL"], P[:, 1] - am["cA"]))
    rad = np.hypot(P[:, 1] - am["cA"], P[:, 2] - am["cL"])
    ok = np.isfinite(am["R"])
    Ra = np.interp(ang, am["bins"][ok], am["R"][ok], left=np.nan, right=np.nan)
    return (loc[:, 1] > y_min) & np.isfinite(Ra) & (rad < Ra + lateral_max)


def crossings(f, ref, level):
    a, b = f[:, :-1], f[:, 1:]
    c = (a >= level) & (b < level)
    frac = np.where(c, (a - level) / np.where(a - b == 0, 1, a - b), np.nan)
    tc = np.where(c, T[:-1] + frac * 0.025, np.nan)
    near = np.full(len(f), np.nan)
    dist = np.abs(tc - ref[:, None]); dist[~c] = np.inf
    j = np.argmin(dist, axis=1); hit = np.isfinite(dist[np.arange(len(f)), j])
    near[hit] = tc[np.arange(len(f)), j][hit]
    return near


def field(case, post):
    outp = OUT / f"BGOFFSET_FIELD_{case}_{post}_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    t0 = time.time()
    if post == "frozen":
        tag = ANALYSES["original"][case]
    elif post == "rigid":
        tag = "rigid_height_metalctl" if (SFXD / case / "measure_rigid_height_metalctl.json").exists() else "rigid_height"
    else:
        tag = ANALYSES["revised"][case]
    M = J(SFXD / case / f"measure_{tag}.json")
    Rres = J(ROOT / f"analysis/{case}/results.json")
    fr = geom.LocalFrame.from_dict(Rres["frame"])
    ge = M["readers"]["A"]["graft_extent"]; ext = (float(ge["x_from_mm"]), float(ge["x_to_mm"]))
    img = sitk.ReadImage(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"), sitk.sitkFloat32)
    pre = sitk.GetArrayFromImage(img); vx = float(img.GetSpacing()[0]); del img
    post_a = sitk.GetArrayFromImage(sitk.ReadImage(str(post_path(case, post)), sitk.sitkFloat32))
    fov = ndi.binary_erosion((post_a > -999.0) & (pre > -1150.0), iterations=int(round(3.0 / vx)))

    def _sat(v):
        mx = float(v.max()); n = int((v >= mx - 1.0).sum())
        return (v >= mx - 1.0) if n > 1000 else np.zeros(v.shape, bool)
    metal = ndi.binary_dilation((pre > 7000) | (post_a > 7000) | _sat(pre) | _sat(post_a), iterations=int(round(2.0 / vx)))
    am = arch_model(Rres, pre, vx, fov=fov)
    fpre = ndi.gaussian_filter(pre, geom.SMOOTH_SIGMA); fpost = ndi.gaussian_filter(post_a, geom.SMOOTH_SIGMA)
    del post_a
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
    tpre = np.empty(len(P)); tpost = np.empty(len(P)); inmax = np.empty(len(P))
    for s in range(0, len(P), 15000):
        p, n = P[s:s + 15000], N[s:s + 15000]
        Q = p[:, None, :] + (T[None, :, None] / vx) * n[:, None, :]
        fa = ndi.map_coordinates(fpre, Q.reshape(-1, 3).T, order=1).reshape(len(p), len(T))
        fb = ndi.map_coordinates(fpost, Q.reshape(-1, 3).T, order=1).reshape(len(p), len(T))
        tp = crossings(fa, np.zeros(len(p)), L0)
        tpre[s:s + 15000] = tp
        tpost[s:s + 15000] = crossings(fb, np.nan_to_num(tp), L0)
        ins = (T < 0) & (T > -1.5)
        inmax[s:s + 15000] = ndi.map_coordinates(pre, Q[:, ins].reshape(-1, 3).T, order=1).reshape(len(p), ins.sum()).max(1)
    del fpre, fpost, pre
    Pmm = P * vx; tooth = inmax > 3000
    radp = np.hypot(Pmm[:, 1] - am["cA"], Pmm[:, 2] - am["cL"])
    radial = np.stack([np.zeros(len(P)), (Pmm[:, 1] - am["cA"]) / radp, (Pmm[:, 2] - am["cL"]) / radp], 1)
    facing = np.sum(N * radial, 1)
    d = tpost - tpre
    loc = fr.to_local(Pmm)
    graftseg = (loc[:, 0] > ext[0] - 8) & (loc[:, 0] < ext[1] + 8) & (loc[:, 2] > -5) & (loc[:, 2] < 60)
    MAXI = maxilla_keep(fr, Pmm, am)
    far = ~graftseg & np.isfinite(d) & (np.abs(d) < 2.0) & MAXI
    c0 = Pmm[far].mean(0)
    r = Pmm[far] - c0; cr = np.cross(r, N[far])
    Am = np.column_stack([np.sum(N[far] * r, 1), N[far][:, 0], N[far][:, 1], N[far][:, 2], cr[:, 0], cr[:, 1], cr[:, 2],
                          tooth[far].astype(float), (~tooth[far]).astype(float)])
    y = d[far]; w = np.ones(len(y))
    for _ in range(20):
        sw = np.sqrt(w); beta, *_ = np.linalg.lstsq(Am * sw[:, None], y * sw, rcond=None)
        res_ = y - Am @ beta; s = 1.4826 * np.median(np.abs(res_)); u = res_ / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

    def disp(Pm):
        rr_ = Pm - c0
        return beta[0] * rr_ + beta[1:4] + np.cross(beta[4:7], rr_)
    eb = fr.R[2]
    readers = {}
    for R in READERS:
        resR = Rres if R == "A" else J(ROOT / f"analysis/{case}/readerB/results.json")
        frR = geom.LocalFrame.from_dict(resR["frame"])
        locR = frR.to_local(Pmm); ebR = frR.R[2]
        boxes = {"window": (resR["roi"]["X_mm"], resR["roi"]["Y_mm"])}
        for nm, wv in M["readers"][R]["controls"]["windows"].items():
            if wv.get("Y_mm"):
                boxes[f"control_{nm}"] = (wv["X_mm"], wv["Y_mm"])
        rows = {}
        for nm, (xr, yr) in boxes.items():
            m = ((locR[:, 0] > xr[0]) & (locR[:, 0] < xr[1]) & (locR[:, 1] > yr[0]) & (locR[:, 1] < yr[1])
                 & (facing > 0.3) & (N @ ebR > 0.5) & np.isfinite(d))
            if m.sum() < 20:
                rows[nm] = {"n_points": int(m.sum())}
                continue
            pred_n = np.sum(disp(Pmm[m]) * N[m], 1)
            rows[nm] = {"n_points": int(m.sum()), "observed_normal_displacement_median_mm": round(float(np.median(d[m])), 3),
                        "global_model_normal_displacement_median_mm": round(float(np.median(pred_n)), 3),
                        "global_model_buccal_displacement_median_mm": round(float(np.median(disp(Pmm[m]) @ ebR)), 3),
                        "local_part_median_mm": round(float(np.median(d[m] - pred_n)), 3)}
        readers[R] = rows
    out = {"_what": "", "case": case, "paper_code": PAPER[case], "post": post, "control_tag": tag,
           "post_file": str(post_path(case, post).relative_to(ROOT)), "n_far_points": int(far.sum()),
           "model": {k: round(float(beta[i]), 5) for i, k in enumerate(["sigma", "t_S", "t_A", "t_L", "w_S", "w_A", "w_L",
                                                                       "delta_tooth", "delta_bone"])},
           "far_median_abs_residual_mm": round(float(np.median(np.abs(y - Am @ beta))), 3),
           "control_and_window_footprints": readers,
           "inputs_sha256": {str(post_path(case, post).relative_to(ROOT)): sha256_file(post_path(case, post))},
           "script_sha256": sha256_file(Path(__file__).resolve()), "peak_rss_GiB": round(rss_gib(), 2),
           "runtime_s": round(time.time() - t0, 1), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    print(json.dumps({k: out[k] for k in ("case", "post", "model", "control_and_window_footprints")}, indent=1))


def collect():
    outp = OUT / f"BACKGROUND_OFFSET_EFFECT_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    Vv = J(OUT / f"BGOFFSET_VOLUMES_{TAG}.json")
    fields = {p.name: J(p) for p in sorted(OUT.glob(f"BGOFFSET_FIELD_*_{TAG}.json"))}
    out = {"_what": "",
           "volumes_file": f"BGOFFSET_VOLUMES_{TAG}.json", "gate_passed": Vv["gate_passed"], "pairs": Vv["pairs"],
           "cases": {an: {cc: {R: {k: r[k] for k in ("background_mm", "offset_nominal_mm3", "offset_effective_mm3_pin_corrected",
                                                     "as_executed_net_mm3", "as_executed_net_without_offset_mm3",
                                                     "pin_corrected_net_mm3", "pin_corrected_net_without_offset_mm3",
                                                     "combined_split", "measured_area_mm2")}
                                   for R, r in v["readers"].items()} for cc, v in a.items()} for an, a in Vv["cases"].items()},
           "S1_offset_by_registration_variant": Vv["offset_by_registration_variant"]["S1"],
           "S1_similarity_fit": Vv["S1_similarity_fit"],
           "global_field_tests": {k: {kk: v[kk] for kk in ("case", "post", "control_tag", "model", "far_median_abs_residual_mm",
                                                           "control_and_window_footprints", "peak_rss_GiB")}
                                  for k, v in fields.items()},
           "inputs_sha256": {f"BGOFFSET_VOLUMES_{TAG}.json": sha256_file(OUT / f"BGOFFSET_VOLUMES_{TAG}.json"),
                             **{k: sha256_file(OUT / k) for k in fields}},
           "script_sha256": sha256_file(Path(__file__).resolve()), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    write_new(outp, out)
    L = ["| analysis | case | assessor | offset mm (nominal mm³) | split rigid / dilation mm | net with / without offset, pin-corrected | as executed with / without |",
         "|---|---|---|---|---|---|---|"]
    for an, a in out["cases"].items():
        for cc, rr in a.items():
            for R, r in rr.items():
                sp = r["combined_split"] or {}
                L.append(f"| {an} | {cc} | {R} | {r['background_mm']:+.3f} ({r['offset_nominal_mm3']:+.1f}) | "
                         f"{sp.get('rigid_shift_mm')} / {sp.get('surface_dilation_mm')} | {r['pin_corrected_net_mm3']} / "
                         f"{r['pin_corrected_net_without_offset_mm3']} | {r['as_executed_net_mm3']} / {r['as_executed_net_without_offset_mm3']} |")
    L += ["", "| analysis | assessor basis | X1 − S1 with offset | without offset | order without offset |", "|---|---|---|---|---|"]
    for an, p in out["pairs"].items():
        for R in READERS:
            for basis in ("pin_corrected", "as_executed"):
                a_, b_ = p[f"{R}_{basis}_with_offset"], p[f"{R}_{basis}_without_offset"]
                L.append(f"| {an} | {R} {basis} | {a_['X1_minus_S1_mm3']:+.1f} | {b_['X1_minus_S1_mm3']:+.1f} | {b_['implant_pair_order']} |")
    with open(OUT / f"BACKGROUND_OFFSET_EFFECT_{TAG}.md", "x") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--volumes"]:
        volumes()
    elif len(a) == 3 and a[0] == "--field" and a[1] in PAPER:
        field(a[1], a[2])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit("usage: --volumes | --field <case> <post> | --collect")

