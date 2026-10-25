#!/usr/bin/env python3
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "analysis"))
from pipeline import geom

OUT = Path(__file__).resolve().parent
TAG = "2026-10-24"
PAPER = {"S1": "S1", "S3": "X1", "S2": "S2", "S4": "X2"}
DEPTHS = (1.0, 3.0, 5.0)
LEVEL = 450.0
STEP = 0.02
REACH = 12.0


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def line_profile(field, vx, fr, X, Y, z_lo, z_hi):
    z = np.arange(z_lo, z_hi + 1e-9, STEP)
    P = np.column_stack([np.full_like(z, X), np.full_like(z, Y), z])
    S = fr.to_anatomical(P) / vx
    ok = np.all((S >= 0) & (S <= np.array(field.shape) - 1), axis=1)
    g = np.full(len(z), np.nan)
    g[ok] = ndi.map_coordinates(field, S[ok].T, order=1)
    return z, g


def surfaces(z, g):
    if not np.all(np.isfinite(g)):
        return None, None, "line leaves the scan"
    above = g >= LEVEL
    if not above.any():
        return None, None, "no bone on the line"
    if above[0] or above[-1]:
        return None, None, "bone reaches the end of the search line"
    ib = np.max(np.nonzero(above)[0]); ip = np.min(np.nonzero(above)[0])
    zb = z[ib] + (g[ib] - LEVEL) / (g[ib] - g[ib + 1]) * STEP
    zp = z[ip - 1] + (LEVEL - g[ip - 1]) / (g[ip] - g[ip - 1]) * STEP
    return float(zb), float(zp), None


def axis_x(lm_loc, a, b, Y):
    p, q = lm_loc[a], lm_loc[b]
    if not (min(p[1], q[1]) <= Y <= max(p[1], q[1])):
        return None
    t = (Y - p[1]) / (q[1] - p[1])
    return float(p[0] + t * (q[0] - p[0]))


def run(case):
    outp = OUT / f"DEFECT_DIMS_{case}_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    vol, vx = geom.load_volume(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"))
    field = ndi.gaussian_filter(vol, geom.SMOOTH_SIGMA); del vol
    rec = {"case": case, "paper_code": PAPER[case], "voxel_mm": vx, "readers": {}}
    for R in ("A", "B"):
        rp = ROOT / (f"analysis/{case}/results.json" if R == "A" else f"analysis/{case}/readerB/results.json")
        res = json.loads(rp.read_text())
        fr = geom.LocalFrame.from_dict(res["frame"])
        lm = res["config"]["landmarks_preop_SAL_mm"]
        L = {k: fr.to_local(np.array([lm[k]], float))[0] for k in geom.ANATOMY_LANDMARKS}
        y0 = float(res["roi"]["Y_mm"][0]); xw = [float(v) for v in res["roi"]["X_mm"]]
        xmid = 0.5 * (xw[0] + xw[1])
        z_lo = min(L["crest_lingual_midspan"][2], L["crest_buccal_midspan"][2]) - REACH
        z_hi = max(L["crest_lingual_midspan"][2], L["crest_buccal_midspan"][2]) + REACH
        cm, cd = np.array(lm["mesial_tooth_cej_buccal"], float), np.array(lm["distal_tooth_cej_buccal"], float)
        span = {"buccal_CEJ_to_buccal_CEJ_mm": round(float(np.linalg.norm(cm - cd)), 2),
                "buccal_CEJ_to_buccal_CEJ_along_ridge_mm": round(abs(float(L["mesial_tooth_cej_buccal"][0] - L["distal_tooth_cej_buccal"][0])), 2),
                "root_axes_at_crestal_plane_mm": None}
        xm0 = axis_x(L, "mesial_tooth_cej_buccal", "mesial_tooth_root_apex", y0)
        xd0 = axis_x(L, "distal_tooth_cej_buccal", "distal_tooth_root_apex", y0)
        if xm0 is not None and xd0 is not None:
            span["root_axes_at_crestal_plane_mm"] = round(abs(xm0 - xd0), 2)
        reach = {}
        for nm, a, b in (("mesial", "mesial_tooth_cej_buccal", "mesial_tooth_root_apex"),
                         ("distal", "distal_tooth_cej_buccal", "distal_tooth_root_apex")):
            xs = [(Y, axis_x(L, a, b, Y)) for Y in np.arange(y0, float(res["roi"]["Y_mm"][1]) + 1e-9, 0.5)]
            xs = [(Y, x) for Y, x in xs if x is not None]
            if not xs:
                reach[nm] = {"root_axis_within_window_depth": False}
                continue
            d_edge = [min(abs(x - xw[0]), abs(x - xw[1])) * (-1 if xw[0] < x < xw[1] else 1) for Y, x in xs]
            reach[nm] = {"root_axis_X_mm_at_window_depths": [[round(Y, 1), round(x, 2)] for Y, x in xs],
                         "axis_inside_window_at_any_depth": any(xw[0] < x < xw[1] for Y, x in xs),
                         "min_distance_axis_to_window_edge_mm": round(float(min(d_edge)), 2),
                         "cej_X_mm": round(float(L[a][0]), 2), "apex_X_mm": round(float(L[b][0]), 2),
                         "cej_Y_mm": round(float(L[a][1]), 2), "apex_Y_mm": round(float(L[b][1]), 2)}
        depth = {}
        for d in DEPTHS:
            Y = y0 + d
            z, g = line_profile(field, vx, fr, xmid, Y, z_lo, z_hi)
            zb, zp, why = surfaces(z, g)
            row = {"Y_mm": Y, "midspan": {"buccal_Z_mm": None if zb is None else round(zb, 3),
                                          "palatal_Z_mm": None if zp is None else round(zp, 3),
                                          "width_mm": None if zb is None else round(zb - zp, 2), "flag": why}}
            ws, nflag = [], 0
            for X in np.arange(xw[0] + 0.1, xw[1], 0.2):
                zz, gg = line_profile(field, vx, fr, X, Y, z_lo, z_hi)
                b_, p_, w_ = surfaces(zz, gg)
                if b_ is None:
                    nflag += 1
                else:
                    ws.append(b_ - p_)
            row["window_span_mean_width_mm"] = round(float(np.mean(ws)), 2) if ws else None
            row["window_span_n_lines"] = len(ws); row["window_span_n_undefined"] = nflag
            teeth = {}
            for nm, a, b in (("mesial", "mesial_tooth_cej_buccal", "mesial_tooth_root_apex"),
                             ("distal", "distal_tooth_cej_buccal", "distal_tooth_root_apex")):
                xt = axis_x(L, a, b, Y)
                if xt is None:
                    teeth[nm] = None
                    continue
                zz, gg = line_profile(field, vx, fr, xt, Y, z_lo, z_hi)
                b_, p_, w_ = surfaces(zz, gg)
                teeth[nm] = {"X_mm": round(xt, 2), "buccal_Z_mm": None if b_ is None else round(b_, 3), "flag": w_}
            row["bounding_teeth_buccal_contour"] = teeth
            hd = None
            if zb is not None and all(teeth[k] and teeth[k]["buccal_Z_mm"] is not None for k in teeth):
                (x1, z1), (x2, z2) = [(teeth[k]["X_mm"], teeth[k]["buccal_Z_mm"]) for k in ("mesial", "distal")]
                zl = z1 + (xmid - x1) * (z2 - z1) / (x2 - x1)
                hd = round(zl - zb, 2)
            row["horizontal_deficit_mm"] = hd
            depth[f"{d:g}"] = row
        rec["readers"][R] = {"span": span, "window_X_mm": xw, "crestal_plane_Y_mm": y0, "window_reaches_bounding_teeth": reach,
                             "depths": depth, "crest_buccal_lingual_Z_mm": [round(float(L["crest_buccal_midspan"][2]), 2),
                                                                           round(float(L["crest_lingual_midspan"][2]), 2)],
                             "crest_width_between_crest_landmarks_mm": round(float(np.linalg.norm(
                                 np.array(lm["crest_buccal_midspan"], float) - np.array(lm["crest_lingual_midspan"], float))), 2)}
    rec["vertical_deficit"] = None
    rec["vertical_deficit_reason"] = ("")
    rec["inputs_sha256"] = {f"analysis/{case}/pre_cal.nii.gz": sha(ROOT / f"analysis/{case}/pre_cal.nii.gz"),
                            f"analysis/{case}/results.json": sha(ROOT / f"analysis/{case}/results.json"),
                            f"analysis/{case}/readerB/results.json": sha(ROOT / f"analysis/{case}/readerB/results.json")}
    rec["script_sha256"] = sha(Path(__file__).resolve())
    import resource
    rec["peak_rss_GiB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 2)
    rec["utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(outp, "x") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    for R, r in rec["readers"].items():
        print(case, R, r["span"], {k: (v["midspan"]["width_mm"], v["window_span_mean_width_mm"], v["horizontal_deficit_mm"],
                                       v["midspan"]["flag"]) for k, v in r["depths"].items()},
              {k: (v.get("axis_inside_window_at_any_depth"), v.get("min_distance_axis_to_window_edge_mm")) for k, v in r["window_reaches_bounding_teeth"].items()})


def collect():
    outp = OUT / f"DEFECT_DIMENSIONS_PREOP_{TAG}.json"
    if outp.exists():
        raise SystemExit(f"{outp} exists")
    per = {c: json.loads((OUT / f"DEFECT_DIMS_{c}_{TAG}.json").read_text()) for c in ("S1", "S3", "S2", "S4")}
    table = {}
    for c, d in per.items():
        rows = {}
        for R in ("A", "B"):
            r = d["readers"][R]
            rows[R] = {"span_buccal_CEJ_to_buccal_CEJ_mm": r["span"]["buccal_CEJ_to_buccal_CEJ_mm"],
                       "span_root_axes_at_crestal_plane_mm": r["span"]["root_axes_at_crestal_plane_mm"],
                       "residual_width_midspan_mm": {k: v["midspan"]["width_mm"] for k, v in r["depths"].items()},
                       "residual_width_window_mean_mm": {k: v["window_span_mean_width_mm"] for k, v in r["depths"].items()},
                       "horizontal_deficit_mm": {k: v["horizontal_deficit_mm"] for k, v in r["depths"].items()},
                       "window_contains_a_root_axis": {k: v.get("axis_inside_window_at_any_depth") for k, v in r["window_reaches_bounding_teeth"].items()},
                       "root_axis_to_window_edge_min_mm": {k: v.get("min_distance_axis_to_window_edge_mm") for k, v in r["window_reaches_bounding_teeth"].items()}}
        diff = {}
        for k in ("residual_width_midspan_mm", "horizontal_deficit_mm"):
            diff[k] = {dd: (None if rows["A"][k][dd] is None or rows["B"][k][dd] is None else round(rows["B"][k][dd] - rows["A"][k][dd], 2))
                       for dd in rows["A"][k]}
        table[PAPER[c]] = {"A": rows["A"], "B": rows["B"], "B_minus_A": diff}
    out = {"_what": "",
           "definitions": {"span": "",
                           "residual_width": "",
                           "horizontal_deficit": "",
                           "window_reaches_teeth": "",
                           "vertical_deficit": "not computed"},
           "table": table, "per_case_files": {PAPER[c]: f"DEFECT_DIMS_{c}_{TAG}.json" for c in per},
           "peak_rss_GiB": {PAPER[c]: per[c]["peak_rss_GiB"] for c in per},
           "script_sha256": sha(Path(__file__).resolve()), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(outp, "x") as fh:
        fh.write(json.dumps(out, indent=1) + "\n")
    print(json.dumps(table, indent=1)[:6000])


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "--case" and a[1] in PAPER:
        run(a[1])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit("usage: --case S1|S2|S3|S4 | --collect")

