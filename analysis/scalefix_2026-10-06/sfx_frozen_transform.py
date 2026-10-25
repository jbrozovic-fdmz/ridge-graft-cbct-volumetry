#!/usr/bin/env python3
import sys, json
from pathlib import Path
import numpy as np, SimpleITK as sitk
ROOT = Path.cwd(); SFX = ROOT / "analysis/scalefix_2026-10-06"
case = sys.argv[1]
reg = json.loads((ROOT / f"analysis/{case}/registration.json").read_text())
fixed = sitk.ReadImage(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"), sitk.sitkFloat32)
moving = sitk.ReadImage(str(ROOT / f"analysis/{case}/post_cal.nii.gz"), sitk.sitkFloat32)
p = reg["parameters_rot_rad_transl_mm"]
t = sitk.Euler3DTransform()
if reg.get("initial_transform"):
    e = sitk.ReadTransform(str(ROOT / reg["initial_transform"])).Downcast()
    t.SetCenter(e.GetCenter())
else:
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    t.SetCenter(init.Downcast().GetCenter() if hasattr(init, "Downcast") else init.GetCenter())
t.SetParameters(p)
post = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, t, sitk.sitkLinear, -1000.0, sitk.sitkFloat32))
frz = sitk.GetArrayFromImage(sitk.ReadImage(str(ROOT / f"cbct_blinded/{case}/post_reg.nii.gz"), sitk.sitkFloat32))
pre = sitk.GetArrayFromImage(fixed)
m = (pre > 500) & (frz > -900) & (post > -900)
d = np.abs(post[m] - frz[m])
out = {"case": case, "center": list(t.GetCenter()), "parameters": list(t.GetParameters()),
       "rounding_note": "",
       "bone_voxels_compared": int(m.sum()), "abs_diff_median": float(np.median(d)),
       "abs_diff_p99": float(np.percentile(d, 99)), "abs_diff_max": float(d.max()),
       "grey_rms_frozen_vs_reconstructed": float(np.sqrt((d ** 2).mean()))}
sitk.WriteTransform(t, str(SFX / case / "frozen_transform_reconstructed.tfm"))
(SFX / case / "frozen_transform_reconstructed.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))

