from __future__ import annotations

import hashlib
import importlib.util
import json
import resource
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path.cwd()
SFX = ROOT / "analysis/scalefix_2026-10-06"
sys.path.insert(0, str(ROOT / "analysis"))
from pipeline import geom, volume as V

CASES = ("S1", "S2", "S3", "S4")
READERS = ("A", "B")
RULES_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

PIN_RULE1 = ROOT / "analysis/pin_artefact_rule_2026-09-14.py"
PIN_APPLY1 = ROOT / "analysis/pin_artefact_apply_2026-09-14.py"
PIN_RULE3 = ROOT / "analysis/pin_artefact_rule_v3_2026-09-14.py"
PROM = ROOT / "analysis/pin_prominence_D7e_2026-09-14.py"
D7E_APPLY = ROOT / "analysis/pin_artefact_apply_D7e_2026-09-14.py"

POSTS = {
    "frozen": "cbct_blinded/{c}/post_reg.nii.gz",
    "similarity": "analysis/scalefix_2026-10-06/{c}/post_reg_similarity.nii.gz",
    "rigid": "analysis/scalefix_2026-10-06/{c}/post_reg_rigid.nii.gz",
    "aniso": "analysis/scalefix_2026-10-06/{c}/post_reg_aniso.nii.gz",
    "scalelo": "analysis/scalefix_2026-10-06/{c}/post_reg_scalelo.nii.gz",
    "scalehi": "analysis/scalefix_2026-10-06/{c}/post_reg_scalehi.nii.gz",
    "anisolo": "analysis/scalefix_2026-10-06/{c}/post_reg_anisolo.nii.gz",
    "affine": "analysis/scalefix_2026-10-06/{c}/post_reg_affine.nii.gz",
    "affinelo": "analysis/scalefix_2026-10-06/{c}/post_reg_affinelo.nii.gz",
    "affinehi": "analysis/scalefix_2026-10-06/{c}/post_reg_affinehi.nii.gz",
    "scale1p012": "analysis/scalefix_2026-10-06/{c}/post_reg_scale1p012.nii.gz",
    "anisohi": "analysis/scalefix_2026-10-06/{c}/post_reg_anisohi.nii.gz",
}
TFMS = {
    "similarity": "analysis/scalefix_2026-10-06/{c}/reg_similarity.tfm",
    "rigid": "analysis/scalefix_2026-10-06/{c}/reg_rigid.tfm",
    "aniso": "analysis/scalefix_2026-10-06/{c}/reg_aniso.tfm",
    "scalelo": "analysis/scalefix_2026-10-06/{c}/reg_scalelo.tfm",
    "scalehi": "analysis/scalefix_2026-10-06/{c}/reg_scalehi.tfm",
    "anisolo": "analysis/scalefix_2026-10-06/{c}/reg_anisolo.tfm",
    "affine": "analysis/scalefix_2026-10-06/{c}/reg_affine.tfm",
    "affinelo": "analysis/scalefix_2026-10-06/{c}/reg_affinelo.tfm",
    "affinehi": "analysis/scalefix_2026-10-06/{c}/reg_affinehi.tfm",
    "scale1p012": "analysis/scalefix_2026-10-06/{c}/reg_scale1p012.tfm",
    "anisohi": "analysis/scalefix_2026-10-06/{c}/reg_anisohi.tfm",
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def write_new(p, obj):
    p = Path(p)
    with open(p, "x") as fh:
        fh.write(json.dumps(obj, indent=1, default=_np_default) + "\n")


def _np_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def results(case, R):
    p = ROOT / (f"analysis/{case}/results.json" if R == "A" else f"analysis/{case}/readerB/results.json")
    return json.loads(p.read_text())


def post_path(case, post):
    return ROOT / POSTS[post].format(c=case)


class PreGrid:

    def __init__(self, case):
        r = sitk.ImageFileReader()
        r.SetFileName(str(ROOT / f"analysis/{case}/pre_cal.nii.gz"))
        r.ReadImageInformation()
        self.img = sitk.Image([1, 1, 1], sitk.sitkUInt8)
        self.img.SetSpacing(r.GetSpacing()); self.img.SetOrigin(r.GetOrigin()); self.img.SetDirection(r.GetDirection())
        self.vx = float(r.GetSpacing()[0])

    def sal_to_phys(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return np.array([self.img.TransformContinuousIndexToPhysicalPoint((p[2] / self.vx, p[1] / self.vx, p[0] / self.vx))
                         for p in P])

    def phys_to_sal(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        out = []
        for x in X:
            ci = self.img.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in x))
            out.append([ci[2] * self.vx, ci[1] * self.vx, ci[0] * self.vx])
        return np.array(out)


def frozen_tfm(case):
    return sitk.ReadTransform(str(SFX / case / "frozen_transform_reconstructed.tfm")).Downcast()


def new_tfm(case, post):
    return sitk.ReadTransform(str(ROOT / TFMS[post].format(c=case))).Downcast()


def map_frozen_to_new(case, post, P_sal):
    if post == "frozen":
        return np.atleast_2d(np.asarray(P_sal, float)).copy()
    g = PreGrid(case)
    tf, tn = frozen_tfm(case), new_tfm(case, post)
    A = sitk.AffineTransform(3)
    A.SetMatrix(tn.GetMatrix()); A.SetCenter(tn.GetCenter()); A.SetTranslation(tn.GetTranslation())
    for q in ((0.0, 0.0, 0.0), (40.0, 30.0, 20.0), (-25.0, 60.0, 35.0)):
        assert np.allclose(A.TransformPoint(q), tn.TransformPoint(q), atol=1e-9), "affine copy differs"
    tni = A.GetInverse()
    X = g.sal_to_phys(P_sal)
    Y = np.array([tni.TransformPoint(tf.TransformPoint(tuple(x))) for x in X])
    return g.phys_to_sal(Y)

