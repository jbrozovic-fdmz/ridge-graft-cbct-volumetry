#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import geom
from pipeline.arch import ArchCurve

CASES = ["S1", "S2", "S3", "S4"]
TOL_FIXED_MM = (0.16, 0.32)
GRID = 0.02
SAMPLE_PIX = 0.05
R_PRIMARY, R_SENS = 0.50, 0.32
METAL_PAD = 0.50
OUTWARD_MM = (0.75, 1.50)
NEAREST_SEARCH_MM = 2.0
THR = geom.ISO_LEVEL
ZONE = {"palatal_of_arch_mm": 2.0, "off_cut_mm": 2.0, "apical_of_pre_apex_mm": 3.0}
ZONE_SENS_APICAL = 10.0
OUT = Path("FINAL_ARCH_20260907/linear_vertical_gain.json")
FIG = Path("FINAL_ARCH_20260907/figures/QC_linear_vertical_gain_2026-09-13.png")


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def r3(x):
    return None if x is None else round(float(x), 3)


def oriented_arch(case, res):
    frame = geom.LocalFrame.from_dict(res["frame"])
    aj = json.loads(Path(f"cases/{case}_arch.json").read_text())
    arch = ArchCurve(np.array(aj["arch_points_AL_mm"], float))
    arch.orient_normal_outward(arch.points.mean(axis=0))
    eb = np.array(res["frame"]["rows_e_md_e_apical_e_buccal"][2], float)
    yv = float(np.mean(res["roi"]["Y_mm"]))
    st0 = arch.nearest_station(*frame.to_anatomical(
        np.array([[frame.offset[0], yv, frame.offset[2]]]))[0][1:])
    if float(arch.normal[st0, 0] * eb[1] + arch.normal[st0, 1] * eb[2]) < 0:
        arch.normal *= -1.0
    return frame, arch


def sample(vol, vx, arch, st, u, s):
    p, n = arch.points[st], arch.normal[st]
    idx = np.stack([np.asarray(s) / vx, (p[0] + np.asarray(u) * n[0]) / vx,
                    (p[1] + np.asarray(u) * n[1]) / vx])
    return ndi.map_coordinates(vol, idx, order=1, mode="constant", cval=-1000.0)


def dist_to_poly(X, P):
    X = np.asarray(X, float)
    if len(X) == 0:
        return np.zeros(0)
    d = np.full(len(X), np.inf)
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        ab = b - a
        L = float(ab @ ab)
        t = np.clip(((X - a) @ ab) / L, 0.0, 1.0) if L > 0 else np.zeros(len(X))
        q = a + t[:, None] * ab
        d = np.minimum(d, np.hypot(X[:, 0] - q[:, 0], X[:, 1] - q[:, 1]))
    return d


def resample_boundary(P, step=SAMPLE_PIX):
    C = np.vstack([P, P[:1]])
    out = []
    for a, b in zip(C[:-1], C[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        t = np.arange(n) / n
        out.append(a[None] + t[:, None] * (b - a)[None])
    return np.vstack(out)


def _classify(metal, in_max, out_far):
    if metal:
        return "metal_adjacent"
    if in_max is None or in_max < THR:
        return "soft_tissue_closing_edge"
    if out_far is not None and out_far >= THR:
        return "dense_closing_edge"
    return "imaged_bone_surface"


def apex_check(pt, poly, field, raw, vx, arch, st, r, sgn):
    g = np.arange(-(r + METAL_PAD), r + METAL_PAD + 1e-9, SAMPLE_PIX)
    U, S = np.meshgrid(pt[0] + g, pt[1] + g)
    U, S = U.ravel(), S.ravel()
    rr = np.hypot(U - pt[0], S - pt[1])
    rawv = sample(raw, vx, arch, st, U, S)
    metal_pts = rawv > geom.METAL_LEVEL
    metal = bool(metal_pts.any())
    nearest_metal = float(rr[metal_pts].min()) if metal else None
    disc = rr <= r
    f = sample(field, vx, arch, st, U[disc], S[disc])
    inside = MplPath(poly).contains_points(np.column_stack([U[disc], S[disc]]))
    in_max = float(f[inside].max()) if inside.any() else None
    out_med = float(np.median(f[~inside])) if (~inside).any() else None
    t = np.arange(OUTWARD_MM[0], OUTWARD_MM[1] + 1e-9, SAMPLE_PIX)
    u_ray, s_ray = np.full_like(t, pt[0]), pt[1] - sgn * t
    out_ray = ~MplPath(poly).contains_points(np.column_stack([u_ray, s_ray]))
    fr = sample(field, vx, arch, st, u_ray, s_ray)
    rr_raw = sample(raw, vx, arch, st, u_ray, s_ray)
    out_far = float(np.median(fr[out_ray])) if out_ray.any() else None
    raw_disc = rawv[disc]
    nm_in = inside & (raw_disc <= geom.METAL_LEVEL)
    in_max_raw = float(raw_disc[nm_in].max()) if nm_in.any() else None
    nm_ray = out_ray & (rr_raw <= geom.METAL_LEVEL)
    out_far_raw = float(np.median(rr_raw[nm_ray])) if nm_ray.any() else None
    v1 = ("metal_adjacent" if metal else "soft_tissue_closing_edge"
          if (in_max is None or in_max < THR) else "dense_both_sides"
          if (out_med is not None and out_med >= THR) else "imaged_bone_surface")
    g2 = np.arange(-NEAREST_SEARCH_MM, NEAREST_SEARCH_MM + 1e-9, SAMPLE_PIX)
    U2, S2 = np.meshgrid(pt[0] + g2, pt[1] + g2)
    U2, S2 = U2.ravel(), S2.ravel()
    ins2 = MplPath(poly).contains_points(np.column_stack([U2, S2]))
    f2 = sample(field, vx, arch, st, U2[ins2], S2[ins2])
    hit = f2 >= THR
    near = (float(np.hypot(U2[ins2][hit] - pt[0], S2[ins2][hit] - pt[1]).min())
            if hit.any() else None)
    cls = _classify(metal, in_max, out_far)
    return {"class": cls, "flagged": cls != "imaged_bone_surface",
            "inside_max_bone_field": r3(in_max),
            "coronal_ray_median_bone_field": r3(out_far),
            "nearest_inside_bone_level_sample_mm": r3(near),
            "nearest_titanium_level_sample_mm": r3(nearest_metal),
            "class_if_titanium_samples_ignored_raw_grey": (
                _classify(False, in_max_raw, out_far_raw) if metal else None),
            "class_v1_superseded": v1, "outside_disc_median_bone_field_v1": r3(out_med),
            "n_inside": int(inside.sum()), "n_outside": int((~inside).sum())}


def check_both_radii(pt, poly, field, raw, vx, arch, st, sgn):
    a = apex_check(pt, poly, field, raw, vx, arch, st, R_PRIMARY, sgn)
    b = apex_check(pt, poly, field, raw, vx, arch, st, R_SENS, sgn)
    a["class_at_r032"] = b["class"]
    a["class_changes_with_radius"] = b["class"] != a["class"]
    return a


def zone_distances(P, Q, sgn, apical_mm):
    hP = -sgn * P[:, 1]
    apex_depth = -hP.max()
    umin = max(P[:, 0].min(), Q[:, 0].min()) + ZONE["off_cut_mm"]
    deep = min((sgn * P[:, 1]).max(), (sgn * Q[:, 1]).max()) - ZONE["off_cut_mm"]
    xs = []
    for X, Y in ((resample_boundary(Q), P), (resample_boundary(P), Q)):
        m = ((X[:, 0] <= -ZONE["palatal_of_arch_mm"]) & (X[:, 0] >= umin)
             & (sgn * X[:, 1] >= apex_depth + apical_mm) & (sgn * X[:, 1] <= deep))
        xs.append(dist_to_poly(X[m], Y))
    return np.concatenate(xs)


def summarise(values, noise, stations):
    v = [(s, x) for s, x in zip(stations, values) if x is not None]
    if not v:
        return {"n_defined": 0, "median_mm": None, "max_mm": None,
                "n_exceeding_noise": 0, "stations_exceeding_noise": []}
    arr = np.array([x for _, x in v])
    exc = [s for s, x in v if x > noise]
    return {"n_defined": len(v), "median_mm": r3(np.median(arr)), "max_mm": r3(arr.max()),
            "max_at_station": v[int(arr.argmax())][0],
            "n_exceeding_noise": len(exc), "stations_exceeding_noise": exc}


def run_case(case):
    d = json.loads(Path(f"cases/{case}_outlines.json").read_text())
    res = json.loads(Path(f"analysis/{case}/results.json").read_text())
    frame, arch = oriented_arch(case, res)
    sgn = 1.0 if frame.jaw == "maxilla" else -1.0
    cs = d["pre_op_crest_S_mm"]
    ge = res["graft_extent_SECONDARY"]

    pre_raw, vx = geom.load_volume(f"analysis/{case}/pre_cal.nii.gz")
    post_raw, vx_q = geom.load_volume(f"cbct_blinded/{case}/post_reg.nii.gz")
    assert abs(vx - vx_q) < 1e-6, "pre_cal and post_reg must share a grid"
    pre_f, post_f = geom.bone_field(pre_raw), geom.bone_field(post_raw)
    tols = {"0.16mm": 0.16, "0.32mm": 0.32,
            f"1vox_{vx:.2f}mm": vx, f"2vox_{2 * vx:.2f}mm": 2 * vx}

    rows, qc = [], []
    for k, v in sorted(d["stations"].items(), key=lambda t: int(t[0])):
        st = int(k)
        P = np.array(v["pre_vertices_us_mm"], float)
        Q = np.array(v["post_vertices_us_mm"], float)
        hP, hQ = -sgn * P[:, 1], -sgn * Q[:, 1]
        ip, iq = int(hP.argmax()), int(hQ.argmax())
        pre_apex, post_apex = P[ip], Q[iq]
        crest_change = float(hQ[iq] - hP[ip])

        lo, hi = Q.min(0) - GRID, Q.max(0) + GRID
        U, S = np.meshgrid(np.arange(lo[0], hi[0], GRID), np.arange(lo[1], hi[1], GRID))
        pts = np.column_stack([U.ravel(), S.ravel()])
        cand = pts[MplPath(Q).contains_points(pts) & ~MplPath(P).contains_points(pts)]
        cand = cand[np.argsort(-(-sgn * cand[:, 1]), kind="stable")]
        dc = dist_to_poly(cand, P)
        urban = {}
        for name, tol in tols.items():
            ok = np.nonzero(dc > tol)[0]
            if not len(ok):
                urban[name] = {"value_mm": None, "note": ""}
                continue
            apx = cand[ok[0]]
            urban[name] = {"value_mm": r3(-sgn * apx[1] - hP[ip]),
                           "apex_us_mm": [r3(apx[0]), r3(apx[1])],
                           "apex_distance_outside_pre_mm": r3(dc[ok[0]])}
        f_pre = check_both_radii(pre_apex, P, pre_f, pre_raw, vx, arch, st, sgn)
        f_post = check_both_radii(post_apex, Q, post_f, post_raw, vx, arch, st, sgn)
        for name in urban:
            if urban[name]["value_mm"] is not None:
                urban[name]["apex_check"] = check_both_radii(
                    np.array(urban[name]["apex_us_mm"]), Q, post_f, post_raw, vx, arch, st, sgn)
                urban[name]["clean"] = (not f_pre["flagged"]
                                        and not urban[name]["apex_check"]["flagged"])
        dz = zone_distances(P, Q, sgn, ZONE["apical_of_pre_apex_mm"])
        dz10 = zone_distances(P, Q, sgn, ZONE_SENS_APICAL)
        xloc = float(frame.to_local(np.array([[cs, *arch.points[st]]]))[0][0])
        rows.append({
            "station": st, "arc_length_mm": v["arc_length_mm"],
            "local_X_mm": r3(xloc),
            "inside_automatic_graft_extent": bool(ge["x_from_mm"] <= xloc <= ge["x_to_mm"]),
            "pre_apex_us_mm": [r3(pre_apex[0]), r3(pre_apex[1])],
            "post_apex_us_mm": [r3(post_apex[0]), r3(post_apex[1])],
            "pre_apex_coronal_of_case_crest_scalar_mm": r3(hP[ip] - (-sgn * cs)),
            "apex_u_shift_mm": r3(post_apex[0] - pre_apex[0]),
            "crest_change_mm": r3(crest_change),
            "crest_change_clean": (not f_pre["flagged"]) and (not f_post["flagged"]),
            "pre_apex_check": f_pre, "post_apex_check": f_post,
            "urban": urban,
            "noise_zone": {"n_samples": int(len(dz)),
                           "median_mm": r3(np.median(dz)) if len(dz) else None,
                           "p95_mm": r3(np.percentile(dz, 95)) if len(dz) else None,
                           "p95_mm_zone_10mm_apical": (r3(np.percentile(dz10, 95))
                                                       if len(dz10) else None)},
        })
        qc.append((st, P, Q, pre_apex, post_apex, urban.get("0.16mm"), f_pre, f_post))
        print(f"{case} st{st:>3}  crest {crest_change:+6.2f} [{f_pre['class'][:6]}/"
              f"{f_post['class'][:6]}]  urban0.16 {urban['0.16mm']['value_mm']}  "
              f"zone p95 {rows[-1]['noise_zone']['p95_mm']}", flush=True)

    p95 = [r["noise_zone"]["p95_mm"] for r in rows if r["noise_zone"]["p95_mm"] is not None]
    p95_10 = [r["noise_zone"]["p95_mm_zone_10mm_apical"] for r in rows
              if r["noise_zone"]["p95_mm_zone_10mm_apical"] is not None]
    noise = float(np.median(p95))
    sts = [r["station"] for r in rows]

    def block(noise_mm):
        crest_all = [r["crest_change_mm"] for r in rows]
        crest_clean = [r["crest_change_mm"] if r["crest_change_clean"] else None for r in rows]
        crest_flag = [None if r["crest_change_clean"] else r["crest_change_mm"] for r in rows]
        o = {"noise_mm": r3(noise_mm),
             "crest_change": {"all": summarise(crest_all, noise_mm, sts),
                              "unflagged": summarise(crest_clean, noise_mm, sts),
                              "flagged": summarise(crest_flag, noise_mm, sts)}}
        for name in tols:
            ua = [r["urban"][name]["value_mm"] for r in rows]
            uc = [r["urban"][name]["value_mm"] if r["urban"][name].get("clean") else None
                  for r in rows]
            uf = [r["urban"][name]["value_mm"] if (r["urban"][name]["value_mm"] is not None
                                                   and not r["urban"][name]["clean"]) else None
                  for r in rows]
            o[f"urban_{name}"] = {"all": summarise(ua, noise_mm, sts),
                                  "unflagged": summarise(uc, noise_mm, sts),
                                  "flagged": summarise(uf, noise_mm, sts)}
        return o

    out = {"voxel_mm": r3(vx), "tolerances_mm": {k: r3(t) for k, t in tols.items()},
           "stations": rows,
           "noise": {"primary_mm": r3(noise),
                     "definition": "",
                     "per_station_p95_mm": p95,
                     "sensitivity_zone_10mm_apical_mm": r3(np.median(p95_10)) if p95_10 else None},
           "summary_primary_noise": block(noise),
           "summary_zone10_noise": block(float(np.median(p95_10))) if p95_10 else None,
           "inputs_sha256_16": {p: sha16(p) for p in (f"cases/{case}_outlines.json",
                                                      f"cases/{case}_arch.json",
                                                      f"analysis/{case}/results.json")}}
    panels = []
    for st, P, Q, pa, qa, ub, fp, fq in qc:
        s_mid = (pa[1] + qa[1]) / 2
        img, ext, _, _ = arch.section(post_f, vx, st, half_width_mm=9.0,
                                      s_range=(s_mid - 6.0, s_mid + 6.0), pixel_mm=0.1)
        panels.append((st, img, ext, P, Q, pa, qa, ub, fp, fq))
    del pre_raw, post_raw, pre_f, post_f
    return out, panels


COL = {"imaged_bone_surface": "lime", "soft_tissue_closing_edge": "red",
       "dense_closing_edge": "magenta", "metal_adjacent": "yellow"}


def main():
    cases = [c for c in sys.argv[1:] if not c.startswith("-")] or CASES
    result = {
        "_what": "",
        "_script": "analysis/linear_vertical_gain.py",
        "_definitions": {
            "crest_change_mm": "",
            "urban_mm": "",
            "apex_check": f"geom.bone_field vs geom.ISO_LEVEL={THR:g}, disc r={R_PRIMARY} mm "
                          f"(sensitivity {R_SENS} mm); metal_adjacent if raw > "
                          f"{geom.METAL_LEVEL:g} within r+{METAL_PAD} mm (UNVERIFIABLE, not "
                          f"refuted); soft_tissue_closing_edge if max inside-outline sample < "
                          f"threshold; dense_closing_edge if median bone field "
                          f"{OUTWARD_MM[0]}-{OUTWARD_MM[1]} mm coronal of the apex >= threshold. "
                          f"v1 rule (median of outside half-disc) superseded after the first run, "
                          f"kept as class_v1_superseded",
            "clean": "",
            "noise": ZONE | {"statistic": "",
                             "sensitivity_apical_mm": ZONE_SENS_APICAL,
                             "disclosure": ""},
            "exceeds_noise": "",
        },
        "software": {"numpy": np.__version__, "scipy": scipy.__version__,
                     "python": sys.version.split()[0]},
        "cases": {}}
    allpanels = {}
    for c in cases:
        result["cases"][c], allpanels[c] = run_case(c)

    pooled = [p for c in result["cases"].values() for p in c["noise"]["per_station_p95_mm"]]
    pooled_noise = float(np.median(pooled))
    result["noise_pooled_all_stations_mm"] = r3(pooled_noise)

    def beyond_noise(key, noise_of):
        out = {}
        for c, o in result["cases"].items():
            n = noise_of(c)
            ver, flg = [], []
            for r in o["stations"]:
                if key == "crest":
                    val, clean = r["crest_change_mm"], r["crest_change_clean"]
                    cls = [r["pre_apex_check"]["class"], r["post_apex_check"]["class"]]
                else:
                    u = r["urban"][key]
                    val, clean = u["value_mm"], u.get("clean", False)
                    cls = ([r["pre_apex_check"]["class"], u["apex_check"]["class"]]
                           if val is not None else None)
                if val is None or val <= n:
                    continue
                (ver if clean else flg).append(
                    r["station"] if clean else {"station": r["station"], "value_mm": val,
                                                "pre_class/apex_class": cls})
            status = ("verified gain beyond noise" if ver else
                      "gain beyond noise at flagged stations only - unverifiable" if flg else
                      "no station beyond noise")
            out[c] = {"noise_mm": r3(n), "verified_stations": ver, "flagged_stations": flg,
                      "status": status}
        return out

    noises = {"primary": lambda c: result["cases"][c]["noise"]["primary_mm"],
              "zone10": lambda c: result["cases"][c]["noise"]["sensitivity_zone_10mm_apical_mm"],
              "pooled": lambda c: pooled_noise}
    variants = {f"{key}|{nn}": beyond_noise(key, nf)
                for nn, nf in noises.items() for key in ("crest", "0.16mm", "0.32mm")}
    claim = {"S1", "S3"}

    def sets(v):
        anyb = {c for c, x in v.items() if x["verified_stations"] or x["flagged_stations"]}
        ver = {c for c, x in v.items() if x["verified_stations"]}
        return anyb, ver
    prim = variants["0.16mm|primary"]
    any_p, ver_p = sets(prim)
    result["verdict"] = {
        "claim_tested": "",
        "rule": "",
        "per_case": prim,
        "cases_beyond_noise_any_flag": sorted(any_p),
        "cases_beyond_noise_verified": sorted(ver_p),
        "pattern_matches_claim": any_p == claim,
        "claim_verified_by_intensity": ver_p == claim,
        "n_stations_beyond_noise_per_case": {c: len(x["verified_stations"])
                                             + len(x["flagged_stations"])
                                             for c, x in prim.items()},
        "all_variants": variants,
        "variants_whose_any_flag_pattern_matches_claim": sorted(
            k for k, v in variants.items() if sets(v)[0] == claim),
        "variants_whose_verified_pattern_matches_claim": sorted(
            k for k, v in variants.items() if sets(v)[1] == claim),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"written {OUT}")
    print("verdict:", json.dumps(result["verdict"], indent=1))

    fig, axs = plt.subplots(len(allpanels), 5, figsize=(20, 4.2 * len(allpanels)), squeeze=False)
    for i, (c, panels) in enumerate(allpanels.items()):
        for j, (st, img, ext, P, Q, pa, qa, ub, fp, fq) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(img, cmap="gray", vmin=-500, vmax=2000, origin="lower", extent=ext)
            ax.contour(np.linspace(ext[0], ext[1], img.shape[1]),
                       np.linspace(ext[2], ext[3], img.shape[0]), img, levels=[THR],
                       colors="#56B4E9", linewidths=0.5)
            for A, col in ((P, "#0072B2"), (Q, "#E69F00")):
                ax.plot(np.append(A[:, 0], A[0, 0]), np.append(A[:, 1], A[0, 1]), "-",
                        color=col, lw=1.2)
            ax.plot(*pa, "s", mfc="none", mec=COL[fp["class"]], ms=9, mew=2)
            ax.plot(*qa, "o", mfc="none", mec=COL[fq["class"]], ms=9, mew=2)
            if ub and ub["value_mm"] is not None:
                ax.plot(*ub["apex_us_mm"], "x", color=COL[ub["apex_check"]["class"]], ms=9, mew=2)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            r = result["cases"][c]["stations"][j]
            ax.set_title(f"{c} st{st}  crest {r['crest_change_mm']:+.2f}  urban0.16 "
                         f"{ub['value_mm'] if ub else None}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("post_reg bone field; blue = PRE outline, orange = POST; square = pre apex, "
                 "circle = post apex, x = augmented apex; marker colour: lime bone surface, red "
                 "soft-tissue closing edge, magenta dense closing edge, yellow titanium-adjacent "
                 "(unverifiable)",
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(FIG, dpi=80)
    print(f"written {FIG}")


if __name__ == "__main__":
    main()

