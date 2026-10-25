#!/usr/bin/env python3
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfx_common import ROOT, SFX, READERS, geom, load_module, post_path, write_new, rss_gib

os.chdir(ROOT)
CSG = load_module("sfx_csg", ROOT / "analysis/cross_scanner_grey_reverify_2026-09-13.py")
SREG = load_module("sfx_sreg", ROOT / "analysis/site_registration_reverify_2026-09-13.py")
GRID = 0.2


def gain_u(Hpre, H, bg):
    t = H - Hpre - bg
    t = t[np.isfinite(t)]
    return float(t[t > 0].sum() * GRID * GRID)


def halfmax_delta(case, M, post):
    rr = M["readers"]["A"]
    res = json.loads((ROOT / f"analysis/{case}/results.json").read_text())
    frame = geom.LocalFrame.from_dict(res["frame"])
    roi = res["roi"]
    z0, z1 = [float(v) for v in res["search_band_Z_mm"]]
    R = {"roi_graft_NOT_stable": dict(X=roi["X_mm"], Y=roi["Y_mm"], z=(z0, z1), sign=+1, s_lim_z=z1, kind="roi",
                                      stable=False)}
    for nm, w in rr["controls"]["windows"].items():
        if w.get("excluded_by_config"):
            continue
        R[f"control_{nm}_buccal_plate"] = dict(X=w["X_mm"], Y=w["Y_mm"], z=(z0, z1), sign=+1, s_lim_z=z1,
                                               kind="control", stable=bool(w.get("valid")),
                                               frozen_valid=bool(w.get("valid")))
    vx = float(res["controls"]["registration_accuracy"]["voxel_mm"])
    A = {name: {} for name in R}
    for tp, path in (("pre", ROOT / f"analysis/{case}/pre_cal.nii.gz"), ("post", post_path(case, post))):
        samp, _ = CSG.sample_scan(path, frame, R, vx)
        for name, reg in R.items():
            A[name][tp] = CSG.analyse_scan(samp[name], reg["s_lim_z"])
        del samp
    common = {n: A[n]["pre"]["valid"] & A[n]["post"]["valid"] for n in R}
    v0 = CSG.replica(A, R)
    v7 = CSG.replica(A, R, ctrl_key="s_hm", ctrl_keep=common)
    fid = {"background_replica_minus_run_mm": round(v0["background_mm"] - rr["controls"]["background_unrounded"], 4),
           "gross_replica_minus_run_mm3": round(v0["gross_mm3"] - rr["as_executed"]["gain_before_background_mm3"], 1)}
    fid["passes_S2_S4_standard"] = bool(abs(fid["background_replica_minus_run_mm"]) <= 0.05 and
                                        abs(fid["gross_replica_minus_run_mm3"]) <= 0.03 * rr["as_executed"]["gain_before_background_mm3"])
    return {"bg_V0_mm": v0["background_mm"], "bg_V7_mm": v7["background_mm"],
            "delta_mm": round(v7["background_mm"] - v0["background_mm"], 4), "replica_fidelity": fid,
            "V0_controls": v0["controls"], "V7_controls": v7["controls"]}


def tooth_d(case, post):
    vx, _ = SREG.header(ROOT / f"analysis/{case}/pre_cal.nii.gz")
    SREG.CROP_MM = 40.0
    if round(SREG.CROP_MM / vx) * vx > SREG.CROP_MM + 1e-6:
        SREG.CROP_MM = int(np.floor(40.0 / vx + 1e-9)) * vx
    site = SREG.Site(case)
    site.post_path = post_path(case, post)
    sl = site.crop()
    pre = SREG.read_crop(site.pre_path, sl)
    pre_s = ndi.gaussian_filter(pre, geom.SMOOTH_SIGMA)
    pst = SREG.read_crop(site.post_path, sl)
    post_s = ndi.gaussian_filter(pst, geom.SMOOTH_SIGMA)
    axes = site.tooth_axes(pre_s)
    masks, minfo = SREG.build_masks(site, axes, pre, pst, pre_s, post_s)
    fixed, moving = SREG.to_img(pre, site), SREG.to_img(pst, site)
    centre = 0.5 * (axes["mesial"]["cej_centre"] + axes["distal"]["cej_centre"])
    centre[1] = 20.0; centre[2] = 15.0
    tx, info = SREG.register(site, fixed, moving, masks["joint"], centre, (0.0, 0.0, 0.0))
    d = SREG.describe_tx(site, tx, axes)["roi_Z15"]["buccal_mean_mm"]
    return {"d_joint_roi_buccal_mean_mm": d, "mask_voxels": minfo.get("joint_voxels"), "registration": info}


def main():
    case, post, rule = sys.argv[1], sys.argv[2], sys.argv[3]
    tag = f"{post}_{rule}"
    outj = SFX / case / f"members_{tag}.json"
    if outj.exists():
        raise SystemExit(f"{outj} exists")
    t0 = time.time()
    M = json.loads((SFX / case / f"measure_{tag}.json").read_text())
    Z = np.load(ROOT / M["fields_npz"])
    out = {"case": case, "post": post, "ctrl_rule": rule, "measure_json": str((SFX / case / f"measure_{tag}.json").relative_to(ROOT))}
    hm = None
    if case != "S3":
        hm = halfmax_delta(case, M, post)
        out["HALFMAX"] = hm
    else:
        out["HALFMAX"] = {"not_available": ""}
    td = tooth_d(case, post)
    out["TOOTH_REG"] = td
    out["members"] = {}
    for R in READERS:
        rr = M["readers"][R]
        bg = rr["controls"]["background_unrounded"]
        bg4, bfz = round(bg, 4), round(bg, 3)
        Hpre = Z[f"{R}_primary_10x10_Hpre"]; Hc = Z[f"{R}_primary_10x10_d1.4_Hc"]
        imp, metal = rr["as_executed"]["implant_in_envelope_mm3"], rr["primary"]["pin_metal_mm3"]

        def net(bgm):
            g = gain_u(Hpre, Hc, bg4 + (round(bgm, 4) - bfz))
            return round(round(g, 1) - imp - metal, 1)
        mem = {}
        if hm is not None:
            b = bfz + hm["delta_mm"]
            mem["HALFMAX_replica_delta"] = {"background_mm": round(b, 4), "net_mm3": net(b),
                                            "B_transferred_from_A_frame": R == "B"}
        s_ctrl = rr["controls"]["registration_accuracy"]["rigid_shift_mm"]
        b = bfz - s_ctrl + td["d_joint_roi_buccal_mean_mm"]
        mem["TOOTH_REG_joint"] = {"background_mm": round(b, 4), "net_mm3": net(b), "control_rigid_shift_mm": s_ctrl}
        out["members"][R] = mem
        print(case, tag, R, mem, flush=True)
    out["peak_rss_GiB"] = round(rss_gib(), 2); out["runtime_s"] = round(time.time() - t0, 1)
    write_new(outj, out)


if __name__ == "__main__":
    main()

