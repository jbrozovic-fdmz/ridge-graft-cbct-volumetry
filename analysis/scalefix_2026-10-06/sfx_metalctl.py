#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

SFX = Path(__file__).resolve().parent
sys.path.insert(0, str(SFX))
import sfx_measure as SM
from sfx_common import ROOT, post_path, geom, sha256_file, write_new

METAL_HI = 7000.0
Z_LINGUAL_EXT_MM = 2.0
WRAPPER_SHA = sha256_file(Path(__file__).resolve())

case, post, rule = sys.argv[1], sys.argv[2], sys.argv[3]
OFF = "--off" in sys.argv
THR = "--thr" in sys.argv
SUFFIX = "_metalctl_off" if OFF else "_metalctl"
LOG = []
CALL = [0]
METAL_INFO = {}

ORIG_SETUP = SM.setup
ORIG_MC = SM.measure_controls


def setup(case_, post_, depth_json):
    c = ORIG_SETUP(case_, post_, depth_json)
    c.metal = {}
    for which, path in (("pre", ROOT / f"analysis/{case_}/pre_cal.nii.gz"), ("post", post_path(case_, post_))):
        vol, vx = geom.load_volume(str(path))
        vmax = float(vol.max())
        m = vol > METAL_HI
        m |= vol >= vmax - 1.0
        idx = np.argwhere(m).astype(float) * vx
        c.metal[which] = {"SAL_mm": idx, "n_total": int(len(idx)), "volume_max": vmax, "voxel_mm": float(vx),
                          "file": str(Path(path).relative_to(ROOT))}
        METAL_INFO[which] = {k: v for k, v in c.metal[which].items() if k != "SAL_mm"}
        del vol, m
    c.metal_local = {R: {w: (c.frames[R].to_local(c.metal[w]["SAL_mm"]) if len(c.metal[w]["SAL_mm"])
                             else np.zeros((0, 3))) for w in ("pre", "post")} for R in SM.READERS}
    return c


def footprint_count(c, R, xr, yb):
    z0, z1 = c.bands[R]
    out = {}
    for w in ("pre", "post"):
        L = c.metal_local[R][w]
        if not len(L) or yb is None:
            out[w] = 0
            continue
        sel = ((L[:, 0] >= xr[0]) & (L[:, 0] <= xr[1]) & (L[:, 1] >= yb[0]) & (L[:, 1] <= yb[1])
               & (L[:, 2] > z0 - Z_LINGUAL_EXT_MM) & (L[:, 2] < z1))
        out[w] = int(sel.sum())
    return out


def measure_controls(c, R, ctrl, rule_, yband_win, Hpre_win, xe_w, ye_w, **kw):
    CALL[0] += 1
    first = ORIG_MC(c, R, ctrl, rule_, yband_win, Hpre_win, xe_w, ye_w, **kw)
    ev, flagged = {}, {}
    for nm, w in first["windows"].items():
        if w.get("excluded_by_config"):
            ev[nm] = {"X_mm": [float(v) for v in w["X_mm"]], "excluded_by_config": True}
            continue
        yb = w.get("Y_mm")
        cnt = footprint_count(c, R, w["X_mm"], yb)
        ev[nm] = {"X_mm": [float(v) for v in w["X_mm"]], "Y_mm": yb, "metal_voxels": cnt,
                  "valid_before_rule": w.get("valid"), "buccal_median_mm": w.get("buccal_median_mm")}
        if cnt["pre"] + cnt["post"] > 0:
            flagged[nm] = ev[nm]
    LOG.append({"call": CALL[0], "reader": R, "ctrl_rule": rule_, "window_Y_mm": [float(v) for v in yband_win],
                "kwargs": {k: v for k, v in kw.items()}, "windows": ev, "flagged": sorted(flagged),
                "applied": bool(flagged) and not OFF,
                "background_before_rule_mm": first.get("background_unrounded")})
    if OFF or not flagged:
        return first
    saved = c.excluded
    c.excluded = set(saved) | set(flagged)
    try:
        second = ORIG_MC(c, R, ctrl, rule_, yband_win, Hpre_win, xe_w, ye_w, **kw)
    finally:
        c.excluded = saved
    for nm, e in flagged.items():
        second["windows"][nm].update({
            "excluded_by_metal_rule": True,
            "exclusion_reason": "",
            "metal_rule_footprint_Y_mm": e["Y_mm"],
            "metal_voxels_pre_healed": [e["metal_voxels"]["pre"], e["metal_voxels"]["post"]],
            "would_have_read_buccal_median_mm": e["buccal_median_mm"]})
    second["metal_rule"] = {"excluded": sorted(flagged),
                            "background_without_rule_unrounded": first.get("background_unrounded")}
    LOG[-1]["background_after_rule_mm"] = second.get("background_unrounded")
    return second


SM.setup = setup
SM.measure_controls = measure_controls

sys.argv = [SM.__file__, case, post, rule] + (["--thr"] if THR else []) + ["--tag", SUFFIX]
SM.main()

tag = f"{post}_{rule}{SUFFIX}"
side = {"_what": "",
        "rule": "",
        "rule_switched_off": OFF, "case": case, "post": post, "ctrl_rule": rule, "tag": tag,
        "wrapper_sha256_at_start": WRAPPER_SHA, "engine_sha256": sha256_file(Path(SM.__file__).resolve()),
        "metal_totals": METAL_INFO, "calls": LOG}
mj = json.loads((SFX / case / f"measure_{tag}.json").read_text())
side["measure_json"] = str((SFX / case / f"measure_{tag}.json").relative_to(ROOT))
side["n_calls_with_flagged_windows"] = sum(1 for e in LOG if e["flagged"])
side["flagged_summary"] = sorted({(e["reader"], nm, tuple(e["windows"][nm]["X_mm"])) for e in LOG for nm in e["flagged"]})
write_new(SFX / case / f"metalctl_{tag}.json", json.loads(json.dumps(side, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))))
print("METALCTL", case, tag, "calls", len(LOG), "with flagged windows", side["n_calls_with_flagged_windows"],
      "flagged", side["flagged_summary"], flush=True)

