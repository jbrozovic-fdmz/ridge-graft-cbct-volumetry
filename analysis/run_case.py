from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "analysis")
from pipeline import calibrate, figures, geom, landmarks, volume
from pipeline.register import register

GRID_STEP = 0.2
GRID_SWEEP = (0.1, 0.15, 0.2, 0.3, 0.5)
THRESHOLD_SWEEP = (350.0, 450.0, 550.0, 750.0)
MAX_REG_RESIDUAL_MM = 0.90


def log(msg):
    print(f"  {msg}", flush=True)


def run(case, skip_registration=False, anchor=None, reader="A"):
    t0 = time.time()
    base_path = Path(f"cases/{case}.json")
    if not base_path.exists():
        raise SystemExit(f"ABORT: {base_path} does not exist.")
    cfg = json.loads(base_path.read_text())
    cfg_path = base_path
    if reader != "A":
        cfg_path = Path(f"cases/{case}_reader{reader}.json")
        if not cfg_path.exists():
            raise SystemExit(f"ABORT: {cfg_path} does not exist. Reader {reader} has not picked "
                             f"landmarks for {case} yet:\n"
                             f"    PYTHONPATH=pylib python3 analysis/pick_landmarks.py {case} "
                             f"--reader {reader}")
        rcfg = json.loads(cfg_path.read_text())
        cfg["landmarks_preop_SAL_mm"] = rcfg["landmarks_preop_SAL_mm"]
        cfg["reader"] = reader
        cfg["section_mode"] = rcfg.get("section_mode")
        for k in ("_PARTIAL_REPICK", "_APEX_REPICKED", "_superseded_mesial_tooth_root_apex",
                  "_superseded_distal_tooth_root_apex"):
            if k in rcfg:
                cfg[k] = rcfg[k]
    out_dir = Path(f"analysis/{case}" if reader == "A" else f"analysis/{case}/reader{reader}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cc_dir = Path(f"cloudcompare_{case}")
    cc_dir.mkdir(exist_ok=True)
    Path("exports").mkdir(exist_ok=True)

    res = {"case": case, "generated": time.strftime("%Y-%m-%d %H:%M"),
           "config": cfg, "software": software_versions()}

    print(f"\n[{case}] 1. loading, checking orientation, calibrating grey values")
    pre_raw = f"cbct_blinded/{case}/pre.nii.gz"
    post_raw = f"cbct_blinded/{case}/post.nii.gz"
    ori = geom.check_orientation(pre_raw)
    ori_post = geom.check_orientation(post_raw)
    res["orientation_check"] = dict(ori, post=ori_post)
    log(f"numpy axes -> pre {ori['numpy_axis_directions']} post {ori_post['numpy_axis_directions']}"
        f"  {'OK' if ori['matches_assumption'] and ori_post['matches_assumption'] else 'UNEXPECTED — see geom.py'}")
    if not (ori["matches_assumption"] and ori_post["matches_assumption"]):
        raise SystemExit(f"ABORT: {case} is not stored in the (S, A, L) orientation the pipeline "
                         f"assumes; every axis-dependent result would be wrong. See geom.py §1.")

    pre_path = str(out_dir / "pre_cal.nii.gz")
    post_path = str(out_dir / "post_cal.nii.gz")
    cal = {"pre": calibrate.calibrate_image(pre_raw, pre_path),
           "post": calibrate.calibrate_image(post_raw, post_path)}
    cal["post_over_pre_scale_ratio"] = round(cal["post"]["scale"] / cal["pre"]["scale"], 3)
    cal["meaning"] = ("")
    res["calibration"] = cal
    log(f"grey calibration  pre: air {cal['pre']['air_grey']:.0f} soft {cal['pre']['soft_tissue_grey']:.0f} "
        f"x{cal['pre']['scale']:.3f} | post: air {cal['post']['air_grey']:.0f} "
        f"soft {cal['post']['soft_tissue_grey']:.0f} x{cal['post']['scale']:.3f} "
        f"| post/pre {cal['post_over_pre_scale_ratio']:.3f}")

    pre_vol, vx = geom.load_volume(pre_path)
    log(f"voxel {vx:.3f} mm, volume {pre_vol.shape}")

    print(f"[{case}] 2. superimposition")
    reg_out = f"cbct_blinded/{case}/post_reg.nii.gz"
    if skip_registration and Path(reg_out).exists():
        log("re-using existing post_reg.nii.gz")
        post_vol, _ = geom.load_volume(reg_out)
        rj = out_dir / "registration.json"
        reg_info = json.loads(rj.read_text()) if rj.exists() else {
            "method": "",
            "v_initial": {}, "v_final": {}}
    else:
        post_vol, reg_info = register(case, pre_vol, vx, cfg, reg_out,
                                      pre_path=pre_path, post_path=post_path)
    res["registration"] = reg_info
    log(f"stable-bone agreement {reg_info['v_initial'].get('mean_abs_difference')} -> "
        f"{reg_info['v_final'].get('mean_abs_difference')} (lower is better); the authoritative "
        f"accuracy figure is measured in step 5")

    pre_f = geom.bone_field(pre_vol)
    post_f = geom.bone_field(post_vol)

    print(f"[{case}] 3. landmarks")
    mode = anchor or cfg.get("anchor") or "implant"
    if mode not in ("preop_anatomy", "implant"):
        raise SystemExit(f"ABORT: unknown anchor '{mode}'; use 'preop_anatomy' or 'implant'.")

    declared = cfg.get("implant_present_in_post")
    if declared is False:
        imp = {"declared_absent": True}
        implant_ok = False
        log("no implant in this case (declared in cases/%s.json); detection skipped" % case)
    else:
        imp = cfg.get("implant_override") or landmarks.find_implant(post_vol, pre_vol, vx)
        implant_ok = bool(imp and imp.get("present"))
        if declared is True and not implant_ok:
            raise SystemExit(
                f"ABORT: cases/{case}.json declares an implant is present in the post-operative "
                f"scan, but none was detected. Either the declaration is wrong or detection has "
                f"failed; do not report a volume until that is resolved.")
    if implant_ok:
        ends = landmarks.orient_implant(imp, pre_f, post_f, vx)
        imp.update(ends)
        log(f"implant {imp['height_mm']:.1f} mm, {100 * imp['new_in_post_frac']:.0f} % new in post "
            f"({imp['n_candidates']} dense components examined)")
    elif declared is not False:
        log("no implant detected in this scan" + ("" if mode == "preop_anatomy" else
            " -- and the implant anchor needs one"))
        if imp:
            (out_dir / "implant_candidates.json").write_text(
                json.dumps(imp.get("candidates", []), indent=1) + "\n")
            for c in sorted(imp.get("candidates", []), key=lambda c: -c.get("n_voxels", 0))[:8]:
                log(f"  cand n={c.get('n_voxels', 0):6d} h={c.get('height_mm', 0):5.2f} mm "
                    f"new={c.get('new_in_post_frac', 0):.3f}")
        imp = {}

    if mode == "preop_anatomy":
        lm = (cfg.get("landmarks_preop_SAL_mm") or {})
        missing = [k for k in geom.ANATOMY_LANDMARKS if not lm.get(k)]
        if missing:
            raise SystemExit(
                f"ABORT: {case} is anchored on pre-operative anatomy but these landmarks are "
                f"missing from cases/{case}.json: {', '.join(missing)}. Pick them with\n"
                f"    PYTHONPATH=pylib python3 analysis/pick_landmarks.py {case}")
        mid_L, sym, full_arch = geom.mid_sagittal_L_mm(pre_vol, vx)
        if not full_arch:
            log(f"WARNING: only part of the arch is in the field of view, so the mid-sagittal "
                f"estimate is unreliable; the buccal axis falls back to the crest pair")
            mid_L = None
        frame, anat = geom.build_frame_from_anatomy(lm, site_fdi=cfg.get("site_fdi"),
                                                    midline_L_mm=mid_L)
        anat["mid_sagittal"] = {"L_mm": None if mid_L is None else round(mid_L, 2),
                                "symmetry_score": round(sym, 3), "full_arch_in_fov": full_arch}
        res["anchor"] = anat
        sc = anat["buccal_side_check"]
        if "LATERALITY_MISMATCH" in sc:
            log("  LATERALITY: " + sc["LATERALITY_MISMATCH"][:150])
        if "WARNING" in sc:
            log("  WARNING: " + sc["WARNING"][:150])
        log(f"anchor: pre-operative anatomy (teeth {cfg.get('bounding_teeth_fdi')}), "
            f"inter-CEJ span {anat['inter_cej_span_mm']:.2f} mm, "
            f"crest B-P width {anat['crest_bucco_palatal_width_mm']:.2f} mm")
        res["buccal_direction"] = {"e_buccal_SAL": frame.R[2].tolist(),
                                   "source": ""}
        if implant_ok:
            pl_loc = frame.to_local(np.array(imp["platform_SAL_mm"]))[0]
            ax = np.array(imp["apex_SAL_mm"]) - np.array(imp["platform_SAL_mm"])
            ax = ax / np.linalg.norm(ax)
            agree = {"implant_platform_local_XYZ": [round(float(v), 2) for v in pl_loc],
                     "platform_lingual_of_buccal_crest_mm": round(float(15.0 - pl_loc[2]), 2),
                     "md_offset_from_inter_cej_midpoint_mm": round(float(pl_loc[0] - 15.0), 2),
                     "implant_axis_vs_tooth_axis_deg": round(float(np.degrees(np.arccos(
                         abs(np.clip(ax @ frame.R[1], -1, 1))))), 1)}
            res["anchor"]["implant_agreement"] = agree
            log(f"  implant cross-check: platform {agree['platform_lingual_of_buccal_crest_mm']:+.2f} mm "
                f"lingual of the buccal crest, {agree['md_offset_from_inter_cej_midpoint_mm']:+.2f} mm "
                f"from the inter-CEJ midpoint, axes {agree['implant_axis_vs_tooth_axis_deg']:.1f} deg apart")
            if abs(pl_loc[0] - 15.0) > 4.0:
                log(f"  WARNING: the implant is {pl_loc[0] - 15.0:+.2f} mm from the inter-CEJ "
                    f"midpoint; check landmarks 1 and 2 are on the intended teeth")
    else:
        if not implant_ok:
            raise SystemExit(
                f"ABORT: {case} has no implant, so it cannot use the implant anchor. Set "
                f"\"anchor\": \"preop_anatomy\" in cases/{case}.json and pick the six "
                f"pre-operative landmarks (analysis/pick_landmarks.py {case}).")
        platform = np.array(imp["platform_SAL_mm"])
        apex = np.array(imp["apex_SAL_mm"])
        bh = cfg.get("buccal_hint_SAL")
        if bh is None:
            bd = landmarks.find_buccal_direction(pre_f, post_f, imp["centroid_SAL_mm"],
                                                 apex - platform, vx)
            if bd is None:
                raise SystemExit(f"ABORT: could not determine the buccal direction for {case}; "
                                 f"set 'buccal_hint_SAL' in cases/{case}.json.")
            bh = bd["e_buccal_SAL"]
            res["buccal_direction"] = bd
            log(f"buccal direction from the augmentation, concentration {bd['concentration']:.2f}")
        else:
            res["buccal_direction"] = {"e_buccal_SAL": bh, "source": "operator override"}
            log("buccal direction: operator override")
        frame = geom.build_frame(platform, apex, bh)
        res["anchor"] = {"mode": "implant",
                         "definition": (""),
                         "caveat": ("")}
        log("anchor: implant (sensitivity analysis only)")

    if mode == "preop_anatomy":
        from scipy import ndimage as _ndi
        site_w = frame.to_anatomical(np.array([15.0, 20.0, 15.0]))[0]
        zz, yy, xx = np.mgrid[0:pre_vol.shape[0], 0:pre_vol.shape[1], 0:pre_vol.shape[2]]
        near = ((zz * vx - site_w[0]) ** 2 + (yy * vx - site_w[1]) ** 2
                + (xx * vx - site_w[2]) ** 2) < 14.0 ** 2
        gained = _ndi.binary_opening((post_vol > geom.ISO_LEVEL) & (pre_vol <= geom.ISO_LEVEL)
                                     & near, np.ones((3, 3, 3)))
        del zz, yy, xx, near
        nvox = int(gained.sum())
        if nvox > 200:
            lab, nlab = _ndi.label(gained)
            cnts = np.bincount(lab.ravel()); cnts[0] = 0
            big = np.argwhere(lab == int(np.argmax(cnts))).astype(float) * vx
            medz = float(np.median(frame.to_local(big)[:, 2]))
            res["anchor"]["new_bone_check"] = {
                "largest_gained_component_mm3": round(float(cnts.max()) * vx ** 3, 1),
                "median_local_Z_mm": round(medz, 2),
                "buccal_of_crest_plane": bool(medz > 15.0)}
            log(f"  new-bone check: largest gained component "
                f"{cnts.max() * vx ** 3:.0f} mm3 at local Z {medz:.1f} "
                f"({'BUCCAL — sign OK' if medz > 15.0 else 'LINGUAL — SIGN LOOKS INVERTED'})")
            if medz <= 15.0:
                raise SystemExit(
                    f"ABORT: the new bone near this site sits at local Z {medz:.1f}, i.e. LINGUAL "
                    f"of the buccal crest plane (Z=15). A graft is buccal, so the bucco-lingual "
                    f"sign is inverted and the ROI would measure the palatal surface, returning a "
                    f"plausible but far too small volume. Check crest_buccal_midspan and "
                    f"crest_lingual_midspan in cases/{case}.json are not swapped.")
            del lab, big
        del gained

    res["frame"] = frame.as_dict()
    res["implant"] = ({k: v for k, v in imp.items() if not k.startswith("_")}
                      if imp.get("present") else None)
    roi = landmarks.standard_roi(cfg.get("roi_md_mm", 10.0), cfg.get("roi_ap_mm", 10.0),
                                 offset=frame.offset, anchor=mode)
    res["roi"] = roi
    log(f"ROI {roi['size_md_mm']:.0f} x {roi['size_ap_mm']:.0f} mm, "
        f"projected area {roi['projected_area_mm2']:.0f} mm2; jaw = {frame.jaw}; "
        f"+X is {frame.md_positive_is}")

    band_ling = float(cfg.get("search_band_lingual_mm", 4.0))
    band_bucc = float(cfg.get("search_band_buccal_mm", 14.0))
    if mode == "preop_anatomy":
        zl = res["anchor"]["landmarks_local_XYZ_mm"]["crest_lingual_midspan"][2]
        zb = res["anchor"]["landmarks_local_XYZ_mm"]["crest_buccal_midspan"][2]
        z_min, z_max = float(zl) - band_ling, float(zb) + band_bucc
        res["search_band_source"] = ("pre-operative crest landmarks: "
                                     f"{band_ling:.1f} mm lingual of the palatal crest to "
                                     f"{band_bucc:.1f} mm buccal of the buccal crest")
    else:
        imp_local_z = frame.to_local(np.asarray(imp["_voxels"], float) * vx)[:, 2]
        z_min, z_max = float(imp_local_z.min()) - band_ling, float(imp_local_z.max()) + band_bucc
        res["search_band_source"] = "implant body, %.1f mm lingual to %.1f mm buccal" % (
            band_ling, band_bucc)
    res["search_band_Z_mm"] = [round(z_min, 2), round(z_max, 2)]

    md_prof = volume.new_bone_md_profile(pre_f, post_f, frame, vx, z_min, z_max)
    res["new_bone_md_profile"] = md_prof
    extent = volume.graft_extent(md_prof)
    res["graft_extent_SECONDARY"] = extent
    log(f"augmentation extends {extent['extent_mm']:.0f} mm mesiodistally "
        f"(X {extent['x_from_mm']:.0f}..{extent['x_to_mm']:.0f}, peak at {extent['peak_x_mm']:.0f}); "
        f"the standardised ROI spans X {roi['X_mm'][0]:.0f}..{roi['X_mm'][1]:.0f}")

    if cfg.get("control_placement", "clean") == "clean":
        ctrl = landmarks.control_windows_clean(roi, md_prof,
                                               md_positive_is=frame.md_positive_is)
        placement = ctrl.pop("_placement"); rule = ctrl.pop("_rule")
        res["control_placement"] = {"mode": "mean-below-threshold", "rule": rule,
                                    "windows": placement}
        for nm, m in placement.items():
            if m.get("WARNING"):
                log(f"  WARNING ({nm}): {m['WARNING']}")
            else:
                log(f"  control ({nm}) placed {m['gap_mm']:.0f} mm clear of the ROI, "
                    f"mean new bone below the placement threshold")
    else:
        ctrl = landmarks.control_windows(roi)
        res["control_placement"] = {"mode": "fixed-gap", "rule": ""}
    res["control_windows"] = ctrl

    print(f"[{case}] 4. surfaces")
    sl, origin_mm = crop_around(frame, pre_f.shape, vx, half=30.0)
    log(f"cropped to {[s_.stop - s_.start for s_ in sl]} voxels around the site")
    Vp, Fp = geom.iso_surface(pre_f[sl], vx)
    Vq, Fq = geom.iso_surface(post_f[sl], vx)
    Vp = Vp + origin_mm
    Vq = Vq + origin_mm
    geom.write_stl(cc_dir / "pre_cc.stl", frame.to_local(Vp), Fp)
    geom.write_stl(cc_dir / "post_cc.stl", frame.to_local(Vq), Fq)
    KEEP = geom.window_box(roi, ctrl, z_min, z_max)
    pre_pts = geom.surface_points_local(Vp, Fp, frame, KEEP)
    post_pts = geom.surface_points_local(Vq, Fq, frame, KEEP)
    log(f"bucco-lingual search band Z {z_min:.1f} .. {z_max:.1f} mm "
        f"({res['search_band_source']})")
    log(f"pre {len(Fp):,} faces, post {len(Fq):,} faces; STLs written to {cc_dir}/")

    print(f"[{case}] 5. internal negative controls")
    bg_vals, ling_vals, ctrl_detail, ctrl_maps = [], [], {}, []
    background_placeholder = 0.0
    rejected = {}
    excluded = set(cfg.get("exclude_control_windows") or [])
    for nm, xr in ctrl.items():
        if nm in excluded:
            ctrl_detail[nm] = {"valid": False, "excluded_by_config": True,
                               "reason": cfg.get("_exclude_control_windows", "")}
            log(f"control ({nm}) EXCLUDED at case level — see cases/{case}.json")
            continue
        b = volume.measure_window(pre_pts, post_pts, xr, roi["Y_mm"], GRID_STEP, z_min, z_max,
                                  keep_box=KEEP)
        if b.get("cells", 0) < 200:
            log(f"control ({nm}) skipped — only {b.get('cells', 0)} cells with both surfaces")
            continue
        l = volume.measure_window(pre_pts, post_pts, xr, roi["Y_mm"], GRID_STEP, z_min, z_max,
                                  mode="bottom_mean", keep_box=KEEP)
        b_out = b["median_thickness_mm"]
        l_out = -l.get("median_thickness_mm", 0.0) if l.get("cells", 0) >= 200 else None
        entry = {"buccal": {k: v for k, v in b.items() if not k.startswith("_")},
                 "lingual_outward_mm": None if l_out is None else round(l_out, 3)}
        if l_out is not None:
            sh = (b_out - l_out) / 2.0
            di = (b_out + l_out) / 2.0
            entry["rigid_shift_mm"] = round(sh, 3)
            entry["dilation_mm"] = round(di, 3)
            entry["valid"] = bool(abs(sh) <= MAX_REG_RESIDUAL_MM and abs(di) <= MAX_REG_RESIDUAL_MM)
            if not entry["valid"]:
                entry["rejected_because"] = (
                    f"rigid shift {sh:+.2f} mm and dilation {di:+.2f} mm; at least one exceeds "
                    f"{MAX_REG_RESIDUAL_MM} mm. The palatal surface in this window moved "
                    f"{l_out:+.2f} mm while the buccal moved {b_out:+.2f} mm, so the buccal bone "
                    f"here is not undisturbed and cannot serve as a control.")
                rejected[nm] = entry["rejected_because"]
                ctrl_detail[nm] = entry
                log(f"control ({nm}) REJECTED: buccal {b_out:+.2f} mm but palatal {l_out:+.2f} mm "
                    f"(shift {sh:+.2f}, dilation {di:+.2f}) — not undisturbed bone")
                continue
        entry["gate_note"] = (
            f"the tolerance is applied to the window MEDIAN ({b_out:+.3f} mm); the window MEAN "
            f"({b['mean_thickness_mm']:+.3f} mm) is reported as a diagnostic only. Where the two "
            f"diverge the window contains a localised feature -- see the Limitations section. A "
            f"mean-based gate was tested and rejected: it applies a MAD-derived threshold to a "
            f"quantity it was not derived for, and would abort three of the eight runs.")
        ctrl_detail[nm] = entry
        ctrl_maps.append((b["_thickness"] - background_placeholder, b["_x_edges"], b["_y_edges"]))
        bg_vals.append(b_out)
        if l_out is not None:
            ling_vals.append(l_out)
        log(f"control ({nm}): buccal surface {b_out:+.2f} mm, "
            f"lingual surface {0.0 if l_out is None else l_out:+.2f} mm "
            f"over {b['area_mm2']:.0f} mm2")
    if not bg_vals:
        raise SystemExit(
            f"ABORT: no valid control window for {case}. All candidates were rejected because "
            f"their buccal and palatal surfaces disagree, i.e. the buccal bone there changed "
            f"after surgery and is not a control. Details: {rejected}")
    if rejected:
        log(f"  background from {len(bg_vals)} valid window(s); {len(rejected)} rejected")
    background = float(np.mean(bg_vals)) if bg_vals else 0.0
    lingual = float(np.mean(ling_vals)) if ling_vals else None

    shift = dilation = None
    if lingual is not None:
        shift = (background - lingual) / 2.0
        dilation = (background + lingual) / 2.0
    res["controls"] = {
        "windows": ctrl_detail,
        "background_offset_mm": round(background, 3),
        "lingual_offset_mm": None if lingual is None else round(lingual, 3),
        "registration_accuracy": {
            "rigid_shift_mm": None if shift is None else round(shift, 3),
            "surface_dilation_mm": None if dilation is None else round(dilation, 3),
            "voxel_mm": vx,
            "interpretation": ""},
        "note": "",
        "tolerance_mm": MAX_REG_RESIDUAL_MM,
        "tolerance_basis": ("")}
    ctrl_maps = [(T - background, X, Y) for T, X, Y in ctrl_maps]
    log(f"background offset {background:+.3f} mm"
        + (f" | rigid shift {shift:+.3f} mm, dilation {dilation:+.3f} mm" if shift is not None
           else ""))
    bad = {nm: d["buccal"]["median_thickness_mm"] for nm, d in ctrl_detail.items()
           if d.get("valid", True) and abs(d["buccal"]["median_thickness_mm"]) > MAX_REG_RESIDUAL_MM}
    if bad:
        raise SystemExit(
            f"ABORT: internal control window(s) {bad} exceed the {MAX_REG_RESIDUAL_MM} mm limit. "
            f"Undisturbed bone cannot have moved that far, so the superimposition or the buccal "
            f"direction is wrong. Inspect {case} before reporting anything.")
    if len(bg_vals) > 1 and (max(bg_vals) - min(bg_vals)) > 2 * MAX_REG_RESIDUAL_MM:
        raise SystemExit(
            f"ABORT: the two control windows disagree by "
            f"{max(bg_vals) - min(bg_vals):.2f} mm ({bg_vals}). That indicates a rotational "
            f"misregistration, which a single background offset cannot correct.")

    print(f"[{case}] 6. volume")
    main = volume.measure_window(pre_pts, post_pts, roi["X_mm"], roi["Y_mm"], GRID_STEP,
                                 z_min, z_max, background=background, keep_box=KEEP)
    if main["coverage"] < 0.90:
        log(f"WARNING: only {100 * main['coverage']:.0f} % of the ROI has both surfaces inside "
            f"the search band; the remainder had no pre-operative buccal plate and is excluded")
    thickness, xe, ye = main["_thickness"], main["_x_edges"], main["_y_edges"]
    gross = main["gain_mm3"]

    raw = volume.measure_window(None, None, roi["X_mm"], roi["Y_mm"], GRID_STEP, z_min, z_max,
                                background=0.0,
                                fields=(main["_Hpre"], main["_Hpost"], xe, ye))
    sub = volume.implant_volume_in_envelope(imp.get("_voxels"), frame, main["_Hpre"],
                                            main["_Hpost"], xe, ye, vx)
    net_bone = round(gross - sub["implant_in_envelope_mm3"], 1)
    res["implant_subtraction"] = sub
    log(f"gross gain {gross:.1f} mm3; implant inside the envelope "
        f"{sub['implant_in_envelope_mm3']:.1f} mm3; NET BONE GAIN {net_bone:.1f} mm3")

    grid_sens = {}
    for s in GRID_SWEEP:
        if s == GRID_STEP:
            grid_sens[f"{s:.2f}"] = gross
            continue
        grid_sens[f"{s:.2f}"] = volume.measure_window(pre_pts, post_pts, roi["X_mm"],
                                                      roi["Y_mm"], s, z_min, z_max,
                                                      background=background,
                                                      keep_box=KEEP)["gain_mm3"]
    thr_sens = {}
    for lv in THRESHOLD_SWEEP:
        if lv == geom.ISO_LEVEL:
            thr_sens[f"{lv:.0f}"] = gross
            continue
        Vp2, Fp2 = geom.iso_surface(pre_f[sl], vx, level=lv)
        Vq2, Fq2 = geom.iso_surface(post_f[sl], vx, level=lv)
        Vp2 += origin_mm
        Vq2 += origin_mm
        m2 = volume.measure_window(geom.surface_points_local(Vp2, Fp2, frame, KEEP),
                                   geom.surface_points_local(Vq2, Fq2, frame, KEEP),
                                   roi["X_mm"], roi["Y_mm"], GRID_STEP, z_min, z_max,
                                   background=background, keep_box=KEEP)
        thr_sens[f"{lv:.0f}"] = m2["gain_mm3"]
        del Vp2, Fp2, Vq2, Fq2, m2
    log(f"grid sensitivity {list(grid_sens.values())}")
    log(f"threshold sensitivity {list(thr_sens.values())}")

    res["volume"] = {
        "PRIMARY_net_bone_gain_mm3": net_bone,
        "gain_before_implant_subtraction_mm3": gross,
        "gain_before_background_and_implant_mm3": raw["gain_mm3"],
        "loss_mm3": main["loss_mm3"],
        "net_gain_minus_loss_mm3": main["net_mm3"],
        "implant_in_envelope_mm3": sub["implant_in_envelope_mm3"],
        "volume_to_surface_ratio_mm3_per_mm2": round(net_bone / main["area_mm2"], 3),
        "ratio_note": "",
        "mean_thickness_mm": main["mean_thickness_mm"],
        "max_thickness_mm": main["max_thickness_mm"],
        "projected_area_mm2": main["area_mm2"],
        "coverage": main["coverage"],
        "spike_rejection_tol_mm": 2.0,
        "spike_note": "",
        "background_subtracted_mm": round(background, 3),
        "grid_sensitivity_mm3": grid_sens,
        "threshold_sensitivity_mm3": thr_sens,
        "method": "",
        "phantom_accuracy": "",
    }

    print(f"[{case}] 7. independent cross-checks")
    metal = geom.metal_mask(post_vol) if implant_ok else None
    boolean = volume.boolean_volume(pre_f, post_f, frame, roi, vx,
                                    implant_mask=metal,
                                    z_min=z_min, z_max=z_max)
    del pre_f, post_f
    b450 = boolean["450"]
    like = round(raw["gain_mm3"] - sub["implant_in_envelope_mm3"], 1)
    boolean["comparable_primary_mm3"] = like
    boolean["difference_vs_comparable_pct"] = round(
        100 * (b450["gained_opened_mm3"] - like) / max(like, 1e-9), 1)
    boolean["comparison"] = ("")
    res["crosscheck_boolean"] = boolean
    log(f"boolean (voxel count, implant masked, same band): gained "
        f"{b450['gained_opened_mm3']:.1f} mm3, lost {b450['lost_raw_mm3']:.1f} mm3 "
        f"| like-for-like primary {like:.1f} mm3 "
        f"({boolean['difference_vs_comparable_pct']:+.1f} %)")

    max_mode = volume.measure_window(pre_pts, post_pts, roi["X_mm"], roi["Y_mm"], GRID_STEP,
                                     z_min, z_max, background=0.0, mode="max", spike_tol_mm=0.0,
                                     keep_box=KEEP)
    res["python_cloudcompare_equivalent_mm3"] = max_mode["gain_mm3"]
    log(f"Python in CloudCompare's mode (max-Z, no background subtraction): "
        f"{max_mode['gain_mm3']:.1f} mm3")
    cc = cloudcompare_volume(case, roi, GRID_STEP, z_min)
    res["crosscheck_cloudcompare"] = cc
    if cc.get("added_mm3") is not None:
        d = 100 * (cc["added_mm3"] - max_mode["gain_mm3"]) / max(max_mode["gain_mm3"], 1e-9)
        cc["python_equivalent_mm3"] = max_mode["gain_mm3"]
        cc["agreement_with_python_pct"] = round(d, 1)
        cc["comparison"] = ("")
        log(f"CloudCompare 2.5D: {cc['added_mm3']:.1f} mm3 "
            f"({d:+.1f} % vs the Python equivalent)")
    else:
        log("CloudCompare not available — skipped")

    print(f"[{case}] 8. linear gain")
    prof = volume.gain_profile(thickness, xe, ye, roi)
    res["linear_gain"] = {k: v for k, v in prof.items() if not k.startswith("_")}
    lv = prof["levels_mm_apical_to_platform"]
    log(f"gain at 1/2/3/5/7 mm apical: "
        + " ".join(f"{lv[k][0]:.2f}" for k in ("1", "2", "3", "5", "7")) + " mm (mean)")
    log(f"peak {prof['peak']['mean_mm']:.2f} mm mean / {prof['peak']['max_mm']:.2f} max "
        f"at {prof['peak']['depth_mm']:.1f} mm apical")

    print(f"[{case}] 9. figures")
    ra = res["controls"]["registration_accuracy"]
    qc = {"implant": imp, "registration": reg_info,
          "accuracy": {"control distal": ctrl_detail.get("distal", {}).get("buccal", {})
                       .get("median_thickness_mm"),
                       "control mesial": ctrl_detail.get("mesial", {}).get("buccal", {})
                       .get("median_thickness_mm"),
                       "rigid shift": ra["rigid_shift_mm"],
                       "dilation": ra["surface_dilation_mm"]},
          "grid_sensitivity": grid_sens, "threshold_sensitivity": thr_sens,
          "summary_lines": summary_lines(res)}
    made = []
    fig_dir = "exports" if reader == "A" else f"exports/reader{reader}"
    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    disp = volume.measure_window(pre_pts, post_pts,
                                 (roi["X_mm"][0] - 9, roi["X_mm"][1] + 9),
                                 (roi["Y_mm"][0] - 4, roi["Y_mm"][1] + 4),
                                 GRID_STEP, z_min, z_max, keep_box=KEEP)
    del pre_pts, post_pts
    made.append(figures.fig_cross_section(case, pre_vol, post_vol, frame, vx, roi,
                                          thickness, xe, ye,
                                          disp["_Hpre"], disp["_Hpost"],
                                          disp["_x_edges"], disp["_y_edges"],
                                          background=background,
                                          z_lo=z_min, z_hi=z_max,
                                          metal=metal, out_dir=fig_dir))
    del metal
    made.append(figures.fig_gain_map(case, thickness, xe, ye, roi, ctrl,
                                     ctrl_maps=ctrl_maps,
                                     apical_up=frame.apical_is_up_on_screen,
                                     md_positive_is=frame.md_positive_is, out_dir=fig_dir))
    made.append(figures.fig_profiles(case, prof, roi=roi, md_label=f"+X = {frame.md_positive_is}",
                                     apical_up=frame.apical_is_up_on_screen,
                                     md_positive_is=frame.md_positive_is, out_dir=fig_dir))
    try:
        made.append(figures.fig_3d(case, (frame.to_local(Vq), Fq),
                                   disp["_Hpre"], disp["_Hpost"],
                                   disp["_x_edges"], disp["_y_edges"], roi,
                                   implant_vf=implant_surface(imp, frame, vx),
                                   background=background, jaw=frame.jaw,
                                   md_positive_is=frame.md_positive_is, out_dir=fig_dir))
    except Exception as e:
        log(f"3D figure failed: {type(e).__name__}: {e}")
    made.append(figures.fig_qc(case, qc, pre_vol, post_vol, frame, vx, roi, out_dir=fig_dir))
    res["figures"] = made
    for m in made:
        log(f"wrote {m}")

    write_csvs(case, out_dir, res, thickness, xe, ye, prof, roi)
    (out_dir / "results.json").write_text(json.dumps(strip_private(res), indent=2))
    (out_dir / "registration.json").write_text(json.dumps(reg_info, indent=2))
    print(f"[{case}] done in {time.time() - t0:.0f} s -> {out_dir}/results.json")
    return res


def implant_surface(imp, frame, voxel_mm):
    vox = imp.get("_voxels")
    if vox is None or not len(vox):
        return None
    vox = np.asarray(vox)
    lo = vox.min(axis=0) - 2
    hi = vox.max(axis=0) + 3
    m = np.zeros(hi - lo, np.float32)
    m[tuple((vox - lo).T)] = 1000.0
    V, F = geom.iso_surface(ndi.gaussian_filter(m, 0.6), voxel_mm, level=450.0)
    return frame.to_local(V + lo * voxel_mm), F


def crop_around(frame, shape, voxel_mm, half=30.0):
    g = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)])
    corners = frame.to_anatomical(g + 15.0) / voxel_mm
    lo = np.maximum(np.floor(corners.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(corners.max(axis=0)).astype(int) + 1, np.array(shape))
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return sl, lo * voxel_mm


def software_versions():
    import matplotlib
    import scipy
    import skimage
    import SimpleITK as sitk
    v = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
         "scikit-image": skimage.__version__, "SimpleITK": sitk.Version_VersionString(),
         "matplotlib": matplotlib.__version__}
    try:
        import pyvista
        v["pyvista"] = pyvista.__version__
    except Exception:
        pass
    try:
        out = subprocess.run(["CloudCompare", "-SILENT",
                              "-NO_TIMESTAMP"], capture_output=True, text=True, timeout=60)
        for line in (out.stdout or "").splitlines():
            if "CloudCompare v" in line:
                v["CloudCompare"] = line.split("CloudCompare v")[1].split()[0]
                break
    except Exception:
        pass
    return v


def cloudcompare_volume(case, roi, step, z_min=0.0):
    try:
        r = subprocess.run(["./analysis/cc_volume.sh", case,
                            f"{roi['X_mm'][0]}", f"{roi['X_mm'][1]}",
                            f"{roi['Y_mm'][0]}", f"{roi['Y_mm'][1]}", f"{z_min}", f"{step}"],
                           capture_output=True, text=True, timeout=900)
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    out = {"available": True, "raw_report": r.stdout.strip()}
    for line in r.stdout.splitlines():
        if "Added volume" in line:
            out["added_mm3"] = float(line.split(")")[-1])
        elif "Removed volume" in line:
            out["removed_mm3"] = float(line.split(")")[-1])
        elif line.startswith("Surface"):
            out["surface_mm2"] = float(line.split(":")[1])
        elif "Matching cells" in line:
            out["matching_cells"] = line.split(":")[1].strip()
    out["note"] = ("")
    return out


def summary_lines(res):
    v = res["volume"]
    return [f"net bone gain      {v['PRIMARY_net_bone_gain_mm3']:8.1f} mm3",
            f"  before implant   {v['gain_before_implant_subtraction_mm3']:8.1f} mm3",
            f"  implant removed  {v['implant_in_envelope_mm3']:8.1f} mm3",
            f"  loss             {v['loss_mm3']:8.1f} mm3",
            f"vol/surface ratio  {v['volume_to_surface_ratio_mm3_per_mm2']:8.2f} mm3/mm2",
            f"mean thickness     {v['mean_thickness_mm']:8.2f} mm",
            f"max thickness      {v['max_thickness_mm']:8.2f} mm",
            f"background removed {v['background_subtracted_mm']:8.2f} mm",
            f"control offset     {res['controls']['background_offset_mm']:8.2f} mm",
            f"  rigid shift      {_f(res['controls']['registration_accuracy']['rigid_shift_mm'])} mm",
            f"  dilation         {_f(res['controls']['registration_accuracy']['surface_dilation_mm'])} mm",
            f"ROI coverage       {100 * v['coverage']:7.1f} %"]


def _f(x):
    return "     n/a" if x is None else f"{x:8.2f}"


def write_csvs(case, out_dir, res, thickness, xe, ye, prof, roi):
    xc = 0.5 * (xe[:-1] + xe[1:]) - (roi["X_mm"][0] + roi["size_md_mm"] / 2)
    yc = 0.5 * (ye[:-1] + ye[1:]) - roi["Y_mm"][0]
    X, Y = np.meshgrid(xc, yc)
    m = np.isfinite(thickness)
    np.savetxt(out_dir / "gain_map.csv",
               np.column_stack([np.full(m.sum(), 0), X[m], Y[m], thickness[m]]),
               delimiter=",", header="case_index,along_ridge_mm,apical_mm,gain_mm",
               comments="", fmt=["%d", "%.3f", "%.3f", "%.4f"])
    np.savetxt(out_dir / "profile_depth.csv",
               np.column_stack([prof["_depth"], prof["_depth_mean"], prof["_depth_max"]]),
               delimiter=",", header="apical_mm,mean_gain_mm,max_gain_mm", comments="",
               fmt="%.4f")
    np.savetxt(out_dir / "profile_along.csv",
               np.column_stack([prof["_along"], prof["_along_mean"], prof["_along_max"]]),
               delimiter=",", header="along_ridge_mm,mean_gain_mm,max_gain_mm", comments="",
               fmt="%.4f")


def strip_private(o):
    if isinstance(o, dict):
        return {k: strip_private(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, list):
        return [strip_private(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not args:
        raise SystemExit("usage: python3 analysis/run_case.py <CASE> [--skip-registration] "
                         "[--anchor preop_anatomy|implant]")
    anch = None
    if "--anchor" in sys.argv:
        anch = sys.argv[sys.argv.index("--anchor") + 1]
    rdr = "A"
    if "--reader" in sys.argv:
        rdr = sys.argv[sys.argv.index("--reader") + 1].upper()
    run(args[0], skip_registration="--skip-registration" in sys.argv, anchor=anch, reader=rdr)

