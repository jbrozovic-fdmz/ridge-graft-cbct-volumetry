#!/usr/bin/env python3
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfx_common import ROOT, SFX, CASES, post_path, map_frozen_to_new, sha256_file, results
from pipeline import geom
from pipeline.arch import ArchCurve

SCRIPTS = ("null_window_check.py", "linear_vertical_gain.py", "outline_compare.py")


def oriented_arch(case, res):
    frame = geom.LocalFrame.from_dict(res["frame"])
    aj = json.loads((ROOT / f"cases/{case}_arch.json").read_text())
    arch = ArchCurve(np.array(aj["arch_points_AL_mm"], float))
    arch.orient_normal_outward(arch.points.mean(axis=0))
    eb = np.array(res["frame"]["rows_e_md_e_apical_e_buccal"][2], float)
    yv = float(np.mean(res["roi"]["Y_mm"]))
    st0 = arch.nearest_station(*frame.to_anatomical(np.array([[frame.offset[0], yv, frame.offset[2]]]))[0][1:])
    if float(arch.normal[st0, 0] * eb[1] + arch.normal[st0, 1] * eb[2]) < 0:
        arch.normal *= -1.0
    return arch


def map_outlines(case, post):
    d = json.loads((ROOT / f"cases/{case}_outlines.json").read_text())
    arch = oriented_arch(case, results(case, "A"))
    rep = {}
    for k, v in d["stations"].items():
        st = int(k)
        p, n, t = arch.points[st], arch.normal[st], arch.tangent[st]
        Q = np.array(v["post_vertices_us_mm"], float)
        if not len(Q):
            continue
        W = np.column_stack([Q[:, 1], p[0] + Q[:, 0] * n[0], p[1] + Q[:, 0] * n[1]])
        assert np.allclose(W[0], arch.to_world(st, Q[0, 0], Q[0, 1]), atol=1e-9)
        Wn = map_frozen_to_new(case, post, W)
        dAL = Wn[:, 1:] - p[None, :]
        u = dAL @ n
        s = Wn[:, 0]
        oop = dAL @ t
        v["post_vertices_us_mm"] = np.column_stack([u, s]).round(4).tolist()
        disp = np.hypot(u - Q[:, 0], s - Q[:, 1])
        rep[k] = {"n_vertices": int(len(Q)), "in_plane_shift_median_mm": round(float(np.median(disp)), 4),
                  "in_plane_shift_max_mm": round(float(disp.max()), 4),
                  "out_of_plane_median_mm": round(float(np.median(np.abs(oop))), 4),
                  "out_of_plane_max_mm": round(float(np.abs(oop).max()), 4)}
        def _shoelace(P):
            P = np.asarray(P, float)
            return 0.5 * abs(float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(P[:, 1], np.roll(P[:, 0], -1))))
        v["_frozen_post_area_mm2"], v["_frozen_gain_area_mm2"] = v.get("post_area_mm2"), v.get("gain_area_mm2")
        v["post_area_mm2"] = round(_shoelace(v["post_vertices_us_mm"]), 2)
        v["gain_area_mm2"] = round(v["post_area_mm2"] - _shoelace(v["pre_vertices_us_mm"]), 2)
        rep[k]["gain_area_frozen_to_new_mm2"] = [v["_frozen_gain_area_mm2"], v["gain_area_mm2"]]
    d["_scalefix_mapping"] = {"post": post, "what": "", "stations": rep}
    return d


def build(tag, post, results_tag, per_case=None):
    M = SFX / f"mirror_{tag}"
    if M.exists():
        raise SystemExit(f"{M} exists")
    (M / "analysis").mkdir(parents=True)
    os.symlink(ROOT / "analysis/pipeline", M / "analysis/pipeline")
    shas = {}
    for s in SCRIPTS:
        shutil.copy2(ROOT / "analysis" / s, M / "analysis" / s)
        shas[s] = sha256_file(M / "analysis" / s)
        assert shas[s] == sha256_file(ROOT / "analysis" / s)
    (M / "FINAL_ARCH_20260907/figures").mkdir(parents=True)
    (M / "cases").mkdir()
    rep = {}
    post0, rt0 = post, results_tag
    for c in CASES:
        post, results_tag = per_case[c] if per_case else (post0, rt0)
        (M / f"analysis/{c}/readerB").mkdir(parents=True)
        (M / f"cbct_blinded/{c}").mkdir(parents=True)
        os.symlink(ROOT / f"analysis/{c}/pre_cal.nii.gz", M / f"analysis/{c}/pre_cal.nii.gz")
        os.symlink(post_path(c, post), M / f"cbct_blinded/{c}/post_reg.nii.gz")
        if results_tag == "frozen":
            os.symlink(ROOT / f"analysis/{c}/results.json", M / f"analysis/{c}/results.json")
            os.symlink(ROOT / f"analysis/{c}/readerB/results.json", M / f"analysis/{c}/readerB/results.json")
        else:
            shutil.copy2(SFX / c / f"results_{results_tag}_A.json", M / f"analysis/{c}/results.json")
            shutil.copy2(SFX / c / f"results_{results_tag}_B.json", M / f"analysis/{c}/readerB/results.json")
        for f in (f"{c}.json", f"{c}_arch.json"):
            os.symlink(ROOT / "cases" / f, M / "cases" / f)
        if post == "frozen":
            os.symlink(ROOT / f"cases/{c}_outlines.json", M / f"cases/{c}_outlines.json")
        else:
            d = map_outlines(c, post)
            (M / f"cases/{c}_outlines.json").write_text(json.dumps(d, indent=1) + "\n")
            rep[c] = d["_scalefix_mapping"]["stations"]
    (M / "MIRROR.json").write_text(json.dumps({"tag": tag, "post": post0, "results_tag": rt0, "per_case": per_case,
                                               "scripts_sha256": shas, "outline_mapping": rep}, indent=1) + "\n")
    print(M, json.dumps(rep, indent=0)[:2000])


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "build_selection":
        sel = json.loads((SFX / "PRIMARY_SELECTION_2026-10-06.json").read_text())
        pc = {c: (sel["primary_tag"][c].rsplit("_", 1)[0], sel["primary_tag"][c]) for c in CASES}
        build(sys.argv[2], None, None, per_case=pc)

