#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "analysis"))
from pipeline import geom

TAG = "2026-09-14"
CASES = ("S1", "S2", "S3", "S4")
POOLED_CASES = ("S2", "S4")
POST_ONLY_FRAC = 0.9
PIN_GREY_MIN = 5000.0
NEAR_X = (-8.0, 44.0)
NEAR_Y = (7.0, 31.0)
BIN_MM = 0.1
N_BINS = 30
BG_SHELL = (3.0, 5.0)
RUN_BINS = 5
TOL = 1e-6
SENS_STEP = 0.3
MIN_AVAIL_GIB = 6.0
AUDIT_FINAL = ROOT / "FINAL_ARCH_20260907/fixation_pins_2026-09-13.json"
DECL = ROOT / f"FINAL_ARCH_20260907/pin_artefact_rule_DECLARATION_{TAG}.json"
OUT_FINAL = ROOT / f"FINAL_ARCH_20260907/pin_artefact_rule_{TAG}.json"

RULE_TEXT = ""


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_new(p: Path, obj) -> None:
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def wait_for_memory():
    for i in range(61):
        out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        avail = float(out.splitlines()[1].split()[-1])
        print(f"free -g available: {avail} GiB", flush=True)
        if avail >= MIN_AVAIL_GIB:
            return avail
        if i == 60:
            raise SystemExit("memory gate: < 6 GiB available after 60 min")
        time.sleep(60)


def read_crop(path, sl):
    img = sitk.ReadImage(str(path))
    a = sitk.GetArrayViewFromImage(img)
    c = a[sl].astype(np.float32)
    vx = float(img.GetSpacing()[0])
    del a, img
    return c, vx


def pin_components(case, post_c, pre_c, lo, vx):
    res = json.loads((ROOT / f"analysis/{case}/results.json").read_text())
    frame = geom.LocalFrame.from_dict(res["frame"])
    lbl, _ = ndi.label(post_c > geom.METAL_LEVEL)
    objs = ndi.find_objects(lbl)
    pre_metal = pre_c > geom.METAL_LEVEL
    imp = res["implant"]
    implant_label = None
    if imp:
        bmin = (np.array(imp["bbox_min_SAL_vox"]) - lo).tolist()
        bmax = (np.array(imp["bbox_max_SAL_vox"]) - lo).tolist()
        for i, ob in enumerate(objs, 1):
            if ob is not None and [s.start for s in ob] == bmin and [s.stop - 1 for s in ob] == bmax \
                    and int((lbl[ob] == i).sum()) == imp["n_voxels"]:
                implant_label = i
                break
        assert implant_label is not None, "implant component not found"
    pins, rows = [], []
    for i, ob in enumerate(objs, 1):
        if ob is None or i == implant_label:
            continue
        s = lbl[ob] == i
        newf = round(1.0 - float(pre_metal[ob][s].mean()), 3)
        gmax = float(post_c[ob][s].max())
        if newf < POST_ONLY_FRAC or round(gmax) < PIN_GREY_MIN:
            continue
        idx = np.argwhere(s) + np.array([o.start for o in ob]) + lo
        cen = frame.to_local(idx.astype(float) * vx).mean(axis=0)
        if not (NEAR_X[0] <= round(cen[0], 2) <= NEAR_X[1] and NEAR_Y[0] <= round(cen[1], 2) <= NEAR_Y[1]):
            continue
        pins.append(i)
        rows.append({"label": i, "n_voxels": int(s.sum()), "post_grey_max": round(gmax),
                     "new_in_post_frac": newf, "centroid_local_A_XYZ_mm": [round(float(v), 2) for v in cen]})
    inv = json.loads(AUDIT_FINAL.read_text())["inventory_near_site"][case]["pin_candidates"]
    want = sorted((c["label"], c["n_voxels"], round(c["post_grey_max"])) for c in inv)
    got = sorted((r["label"], r["n_voxels"], r["post_grey_max"]) for r in rows)
    assert got == want, f"{case}: pin set {got} != audit inventory {want}"
    return lbl, pins, implant_label, rows


def pin_mask(lbl, pins, vx, d):
    m0 = np.isin(lbl, pins) if pins else np.zeros(lbl.shape, bool)
    if d <= 0 or not m0.any():
        return m0
    return ndi.distance_transform_edt(~m0, sampling=vx) <= d + TOL


def inpaint(post_c, mask):
    if not mask.any():
        return post_c.copy()
    _, ind = ndi.distance_transform_edt(mask, return_indices=True)
    out = post_c[tuple(ind)]
    del ind
    return out


def wquant(v, w, q):
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cw = np.cumsum(w)
    return float(v[np.searchsorted(cw, q * cw[-1], side="left")])


def derive_d(bins_v, bins_w, bg_v, bg_w):
    q1, q3 = wquant(bg_v, bg_w, 0.25), wquant(bg_v, bg_w, 0.75)
    med = [wquant(v, w, 0.5) if len(v) else None for v, w in zip(bins_v, bins_w)]
    inside = [None if m is None else bool(q1 <= m <= q3) for m in med]
    k_star = None
    for k in range(N_BINS):
        if inside[k] is None:
            continue
        run = [inside[j] for j in range(k, min(k + RUN_BINS, N_BINS)) if inside[j] is not None]
        if all(run):
            k_star = k
            break
    d = round(BIN_MM * k_star, 1) if k_star is not None else 3.0
    rows = [{"bin": k, "r_mm": [round(BIN_MM * k, 1), round(BIN_MM * (k + 1), 1)], "n_voxels": int(len(bins_v[k])),
             "post_grey_median": None if med[k] is None else round(med[k], 1), "within_bg_IQR": inside[k]}
            for k in range(N_BINS)]
    return {"d_mm": d, "k_star": k_star, "flag_no_return_within_3mm": k_star is None,
            "background": {"shell_mm": list(BG_SHELL), "n_voxels": int(len(bg_v)), "Q1": round(q1, 1),
                           "median": round(wquant(bg_v, bg_w, 0.5), 1), "Q3": round(q3, 1)},
            "profile": rows}


def bin_of(r):
    return np.ceil((r - TOL) / BIN_MM).astype(int) - 1


def profile(case):
    au = json.loads((ROOT / f"analysis/{case}/fixation_pins_2026-09-13.json").read_text())
    lo = np.array(au["crop_lo_vox"]); hi = np.array(au["crop_hi_vox"])
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    wait_for_memory()
    post_c, vx = read_crop(ROOT / f"cbct_blinded/{case}/post_reg.nii.gz", sl)
    pre_c, vx2 = read_crop(ROOT / f"analysis/{case}/pre_cal.nii.gz", sl)
    assert abs(vx - vx2) < 1e-6 and abs(vx - au["voxel_mm"]) < 1e-6
    print(f"[{case}] crop {post_c.shape} vx {vx:.3f}  RSS {rss_gib():.2f} GiB", flush=True)
    lbl, pins, implant_label, rows = pin_components(case, post_c, pre_c, lo, vx)
    del pre_c
    out = {"case": case, "voxel_mm": vx, "crop_lo_vox": lo.tolist(), "crop_hi_vox": hi.tolist(),
           "implant_label": implant_label, "pins": rows, "rule_text_sha256": sha256_text(RULE_TEXT)}
    npz = {}
    if pins:
        m0 = np.isin(lbl, pins)
        nz = np.argwhere(m0)
        pad = int(np.ceil((2 * BG_SHELL[1] + 0.5) / vx))
        blo = np.maximum(nz.min(0) - pad, 0); bhi = np.minimum(nz.max(0) + pad + 1, np.array(lbl.shape))
        bs = tuple(slice(int(a), int(b)) for a, b in zip(blo, bhi))
        q = post_c[bs]; pm = m0[bs]; metal = q > geom.METAL_LEVEL
        dist, ind = ndi.distance_transform_edt(~metal, sampling=vx, return_indices=True)
        attributed = pm[tuple(ind)]
        del ind, dist
        rpin = ndi.distance_transform_edt(~pm, sampling=vx)
        sel = attributed & ~metal & (rpin <= BG_SHELL[1] + TOL)
        r_sel = rpin[sel].astype(np.float64); g_sel = q[sel].astype(np.float64)
        shell_reach = int(np.ceil((BG_SHELL[1] + TOL) / vx))
        out["profile_box"] = {"lo": (blo + lo).tolist(), "hi": (bhi + lo).tolist(),
                              "pins_bbox_margin_to_crop_edge_vox": int(min((nz.min(0)).min(),
                                                                          (np.array(lbl.shape) - 1 - nz.max(0)).min())),
                              "shell_reach_vox": shell_reach}
        out["n_pin_titanium_voxels"] = int(pm.sum())
        out["pin_titanium_grey_median"] = round(float(np.median(q[pm])), 1)
        out["n_voxels_nearest_to_other_metal_within_5mm_excluded"] = int(
            ((~attributed) & ~metal & (rpin <= BG_SHELL[1] + TOL)).sum())
        npz = {"r": r_sel.astype(np.float32), "grey": g_sel.astype(np.float32)}
        bins_v, bins_w, bg_v, bg_w = split_bins(r_sel, g_sel, np.ones_like(g_sel))
        out["per_case_derivation"] = derive_d(bins_v, bins_w, bg_v, bg_w)
        out["per_case_d_mm_REPORTED_NOT_USED"] = out["per_case_derivation"]["d_mm"]
        print(f"[{case}] per-case d = {out['per_case_d_mm_REPORTED_NOT_USED']} mm  "
              f"bg {out['per_case_derivation']['background']}", flush=True)
    out["peak_rss_GiB"] = round(rss_gib(), 2)
    write_new(ROOT / f"analysis/{case}/pin_bloom_profile_{TAG}.json", out)
    if npz:
        p = ROOT / f"analysis/{case}/pin_bloom_profile_{TAG}.npz"
        assert not p.exists()
        np.savez_compressed(p, **npz)
    print(f"[{case}] written; peak RSS {out['peak_rss_GiB']} GiB", flush=True)


def split_bins(r, g, w):
    k = bin_of(r)
    bins_v = [g[k == j] for j in range(N_BINS)]
    bins_w = [w[k == j] for j in range(N_BINS)]
    bg = (r > BG_SHELL[0] + TOL) & (r <= BG_SHELL[1] + TOL)
    return bins_v, bins_w, g[bg], w[bg]


def declare():
    if DECL.exists():
        raise SystemExit(f"{DECL} exists; not overwritten")
    write_new(DECL, {"_what": "",
                     "declared_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "rule_text": RULE_TEXT, "rule_text_sha256": sha256_text(RULE_TEXT),
                     "script": "analysis/pin_artefact_rule_2026-09-14.py",
                     "script_sha256_at_declaration": sha256_file(Path(__file__).resolve()),
                     "d_mm": None, "window_volumes_computed": False})
    print("declared", sha256_text(RULE_TEXT))


def collect():
    decl = json.loads(DECL.read_text())
    assert decl["rule_text_sha256"] == sha256_text(RULE_TEXT) == sha256_text(decl["rule_text"]), \
        "rule text changed since declaration"
    per = {c: json.loads((ROOT / f"analysis/{c}/pin_bloom_profile_{TAG}.json").read_text()) for c in CASES}
    acc_v = [[] for _ in range(N_BINS)]; acc_w = [[] for _ in range(N_BINS)]; bgv, bgw = [], []
    for c in POOLED_CASES:
        z = np.load(ROOT / f"analysis/{c}/pin_bloom_profile_{TAG}.npz")
        r = z["r"].astype(np.float64); g = z["grey"].astype(np.float64)
        bv, _, b_bg, _ = split_bins(r, g, np.ones_like(g))
        for j in range(N_BINS):
            if len(bv[j]):
                acc_v[j].append(bv[j]); acc_w[j].append(np.full(len(bv[j]), 1.0 / len(bv[j])))
        bgv.append(b_bg); bgw.append(np.full(len(b_bg), 1.0 / len(b_bg)))
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    pooled = derive_d([cat(v) for v in acc_v], [cat(w) for w in acc_w], cat(bgv), cat(bgw))
    d = pooled["d_mm"]
    sens = sorted({round(max(0.0, d - SENS_STEP), 1), d, round(d + SENS_STEP, 1)})
    rule = {
        "_what": "",
        "date": TAG,
        "declaration_file": str(DECL.relative_to(ROOT)),
        "declared_utc": decl["declared_utc"],
        "rule_text": RULE_TEXT,
        "rule_text_sha256": sha256_text(RULE_TEXT),
        "rule_text_sha256_matches_declaration": True,
        "script": "analysis/pin_artefact_rule_2026-09-14.py",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "d_mm": d,
        "sensitivity_d_mm": sens,
        "derivation_pooled": {"cases": list(POOLED_CASES), "case_weighting": "",
                              **pooled},
        "per_case_d_mm_REPORTED_NOT_USED": {c: per[c].get("per_case_d_mm_REPORTED_NOT_USED") for c in CASES},
        "per_case_background": {c: per[c]["per_case_derivation"]["background"]
                                for c in CASES if "per_case_derivation" in per[c]},
        "per_case_derivation": {c: per[c].get("per_case_derivation") for c in CASES},
        "pins": {c: per[c]["pins"] for c in CASES},
        "implant_label": {c: per[c]["implant_label"] for c in CASES},
        "frozen_values_to_reproduce_before_correction": {
            c: {R: v for R, v in json.loads((ROOT / "FINAL_ARCH_20260907/FROZEN_RESULTS.json").read_text())
                ["cases"][c].items() if R in ("A", "B")} for c in CASES},
        "peak_rss_GiB_profile": {c: per[c]["peak_rss_GiB"] for c in CASES},
        "window_volumes_computed": False,
    }
    for c in CASES:
        rule["frozen_values_to_reproduce_before_correction"][c] = {
            R: {"net_mm3": v["net_mm3"], "gross_mm3": v["gross_mm3"]}
            for R, v in rule["frozen_values_to_reproduce_before_correction"][c].items()}
    write_new(OUT_FINAL, rule)
    print(json.dumps({"d_mm": d, "sensitivity": sens, "background": pooled["background"],
                      "k_star": pooled["k_star"],
                      "per_case": rule["per_case_d_mm_REPORTED_NOT_USED"]}, indent=1))
    for row in pooled["profile"][:12]:
        print(row)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--declare"]:
        declare()
    elif len(a) == 2 and a[0] == "--profile" and a[1] in CASES:
        profile(a[1])
    elif a == ["--collect"]:
        collect()
    else:
        raise SystemExit(__doc__)

