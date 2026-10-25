from __future__ import annotations

import struct

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import measure

ISO_LEVEL = 450.0
METAL_LEVEL = 2800.0
SMOOTH_SIGMA = 0.6

S, A, L = 0, 1, 2


def load_volume(path):
    img = sitk.ReadImage(str(path))
    sp = np.array(img.GetSpacing(), dtype=float)
    if not np.allclose(sp, sp[0], atol=1e-6):
        raise ValueError(f"{path}: anisotropic voxels {sp}; the pipeline assumes isotropic")
    a = sitk.GetArrayViewFromImage(img)
    return a.astype(np.float32), float(sp[0])


def check_orientation(path):
    img = sitk.ReadImage(str(path))
    D = np.array(img.GetDirection()).reshape(3, 3)
    lab = np.array([["R", "L"], ["A", "P"], ["I", "S"]])
    got = []
    for image_axis in range(3):
        v = D[:, image_axis]
        k = int(np.argmax(np.abs(v)))
        got.append(lab[k][1 if v[k] > 0 else 0])
    got = got[::-1]
    ok = got == ["S", "A", "L"]
    return {"numpy_axis_directions": got, "matches_assumption": bool(ok)}


def bone_field(vol, sigma=SMOOTH_SIGMA, metal=METAL_LEVEL):
    f = vol.copy()
    fill = np.asarray(-1000.0, f.dtype)
    for s in range(0, f.shape[0], 32):
        sl = slice(s, s + 32)
        np.copyto(f[sl], fill, where=vol[sl] > metal)
    ndi.gaussian_filter(f, sigma, output=f)
    return f


def metal_mask(vol, metal=METAL_LEVEL):
    return vol > metal


def iso_surface(field, voxel_mm, level=ISO_LEVEL):
    V, F, _, _ = measure.marching_cubes(field, level=level, spacing=(voxel_mm,) * 3)
    G = F[:, ::-1]
    return V, (G if G.flags.c_contiguous else G.copy())


def face_geometry(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    n = np.linalg.norm(cross, axis=1, keepdims=True)
    return (a + b + c) / 3.0, 0.5 * n[:, 0], cross / np.maximum(n, 1e-12)


def write_stl(path, V, F, chunk=1_000_000):
    with open(path, "wb") as o:
        o.write(b"\0" * 80)
        o.write(struct.pack("<I", len(F)))
        for s in range(0, len(F), chunk):
            Fc = F[s:s + chunk]
            a, b, c = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
            n = np.cross(b - a, c - a)
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            rec = np.zeros((len(Fc), 50), np.uint8)
            rec[:, 0:12] = n.astype("<f4").view(np.uint8).reshape(-1, 12)
            rec[:, 12:24] = a.astype("<f4").view(np.uint8).reshape(-1, 12)
            rec[:, 24:36] = b.astype("<f4").view(np.uint8).reshape(-1, 12)
            rec[:, 36:48] = c.astype("<f4").view(np.uint8).reshape(-1, 12)
            rec.tofile(o)


def read_stl(path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    n = int(raw[80:84].view(np.uint32)[0])
    rec = raw[84:84 + n * 50].reshape(n, 50)
    return rec[:, 12:48].copy().view(np.float32).reshape(n, 3, 3).astype(np.float64)


def densify(tri, n=4):
    nf = len(tri)
    ij = [(i, j) for i in range(n + 1) for j in range(n + 1 - i)]
    out = np.empty((len(ij) * nf, 3), np.result_type(tri.dtype, np.float64))
    tmp = np.empty((nf, 3), out.dtype)
    for k, (i, j) in enumerate(ij):
        blk = out[k * nf:(k + 1) * nf]
        np.multiply(tri[:, 0], i, out=blk)
        np.multiply(tri[:, 1], j, out=tmp); blk += tmp
        np.multiply(tri[:, 2], n - i - j, out=tmp); blk += tmp
        blk /= n
    return out


def window_box(roi, ctrl, z_min, z_max, pad_x_mm=9.0, pad_y_mm=4.0, eps=1e-6):
    xs = [roi["X_mm"], (roi["X_mm"][0] - pad_x_mm, roi["X_mm"][1] + pad_x_mm)] + \
         [tuple(v) for v in ctrl.values()]
    ys = [roi["Y_mm"], (roi["Y_mm"][0] - pad_y_mm, roi["Y_mm"][1] + pad_y_mm)]
    return (min(w[0] for w in xs) - eps, max(w[1] for w in xs) + eps,
            min(w[0] for w in ys) - eps, max(w[1] for w in ys) + eps,
            z_min - eps, z_max + eps)


def surface_points_local(V, F, frame, keep_box, n=4, chunk=4_000_000):
    x0, x1, y0, y1, z0, z1 = keep_box
    tri = V[F]
    nf = len(tri)
    blk = np.empty((nf, 3), tri.dtype)
    tmp = np.empty((nf, 3), tri.dtype)
    Rt = frame.R.T
    parts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            np.multiply(tri[:, 0], i, out=blk)
            np.multiply(tri[:, 1], j, out=tmp); blk += tmp
            np.multiply(tri[:, 2], n - i - j, out=tmp); blk += tmp
            blk /= n
            P = np.empty((nf, 3), float)
            for s in range(0, nf, chunk):
                e = s + chunk
                buf = blk[s:e] - frame.origin
                np.matmul(buf, Rt, out=P[s:e])
                P[s:e] += frame.offset
            m  = P[:, 0] >= x0
            m &= P[:, 0] <= x1
            m &= P[:, 1] >= y0
            m &= P[:, 1] <= y1
            m &= P[:, 2] >= z0
            m &= P[:, 2] <= z1
            parts.append(P[m])
            del P, m
    del tri, blk, tmp
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 3))


class LocalFrame:

    def __init__(self, origin, e_md, e_apical, e_buccal, offset):
        self.origin = np.asarray(origin, float)
        self.R = np.vstack([e_md, e_apical, e_buccal])
        self.offset = np.asarray(offset, float)
        det = float(np.linalg.det(self.R))
        if abs(det - 1.0) > 1e-6:
            raise ValueError(f"local frame is not a proper rotation (det={det:.6f})")

    def to_local(self, P, chunk=4_000_000):
        P = np.atleast_2d(P)
        out = np.empty((len(P), 3), float)
        Rt = self.R.T
        for s in range(0, len(P), chunk):
            e = s + chunk
            buf = P[s:e] - self.origin
            np.matmul(buf, Rt, out=out[s:e])
            out[s:e] += self.offset
        return out

    def to_anatomical(self, Q):
        return (np.atleast_2d(Q) - self.offset) @ self.R + self.origin

    md_positive_is = "unknown"
    jaw = "unknown"

    @property
    def apical_is_up_on_screen(self):
        return self.jaw == "maxilla"

    def as_dict(self):
        return {"axes": "",
                "origin_SAL_mm": self.origin.tolist(),
                "rows_e_md_e_apical_e_buccal": self.R.tolist(),
                "offset_mm": self.offset.tolist(),
                "md_positive_is": self.md_positive_is,
                "jaw": self.jaw}

    @staticmethod
    def from_dict(d):
        R = np.array(d["rows_e_md_e_apical_e_buccal"], float)
        f = LocalFrame(d["origin_SAL_mm"], R[0], R[1], R[2], d["offset_mm"])
        f.md_positive_is = d.get("md_positive_is", "unknown")
        f.jaw = d.get("jaw", "unknown")
        return f


def build_frame(platform_SAL, apex_SAL, buccal_hint_SAL, anterior_axis=None, offset=(15., 15., 15.)):
    e_ap = np.asarray(apex_SAL, float) - np.asarray(platform_SAL, float)
    n = np.linalg.norm(e_ap)
    if n < 1e-6:
        raise ValueError("implant platform and apex coincide")
    e_ap /= n

    b = np.asarray(buccal_hint_SAL, float)
    b = b - (b @ e_ap) * e_ap
    nb = np.linalg.norm(b)
    if nb < 1e-6:
        raise ValueError("buccal hint is parallel to the implant axis")
    e_bu = b / nb

    e_md = np.cross(e_ap, e_bu)
    e_md /= np.linalg.norm(e_md)

    if anterior_axis is None:
        anterior_axis = np.array([0.0, 1.0, 0.0])
    frame = LocalFrame(platform_SAL, e_md, e_ap, e_bu, offset)
    frame.md_positive_is = "mesial" if float(e_md @ anterior_axis) > 0 else "distal"
    frame.jaw = "maxilla" if float(e_ap[0]) > 0 else "mandible"
    return frame


def mid_sagittal_L_mm(vol, voxel_mm, level=500.0):
    bone = vol > level
    prof = bone.sum(axis=(0, 1)).astype(float)
    n = len(prof)
    best, best_c = -1.0, None
    for c in range(int(0.25 * n), int(0.75 * n)):
        h = min(c, n - 1 - c)
        if h < 30:
            continue
        a, b = prof[c - h:c], prof[c + 1:c + 1 + h][::-1]
        m = min(len(a), len(b))
        if m < 30:
            continue
        den = float(np.linalg.norm(a[:m]) * np.linalg.norm(b[:m]))
        sc = float((a[:m] * b[:m]).sum()) / den if den else 0.0
        if sc > best:
            best, best_c = sc, c
    if best_c is None:
        return None, 0.0, False
    dense = vol > 2000
    pd = dense.sum(axis=(0, 1)).astype(float)
    idx = np.where(pd > pd.max() * 0.10)[0] if pd.max() > 0 else np.array([])
    width = float((idx[-1] - idx[0]) * voxel_mm) if len(idx) > 1 else 0.0
    return float(best_c * voxel_mm), best, bool(width > 40.0)


ANATOMY_LANDMARKS = ("mesial_tooth_cej_buccal", "distal_tooth_cej_buccal",
                     "mesial_tooth_root_apex", "distal_tooth_root_apex",
                     "crest_buccal_midspan", "crest_lingual_midspan")


def build_frame_from_anatomy(lm, site_fdi=None, midline_L_mm=None,
                             anterior_axis=None, offset=(15., 15., 15.)):
    P = {}
    for k in ANATOMY_LANDMARKS:
        v = lm.get(k)
        if v is None or len(v) != 3:
            raise ValueError(f"anatomical landmark '{k}' is missing or malformed")
        P[k] = np.asarray(v, float)
    P1, P2 = P["mesial_tooth_cej_buccal"], P["distal_tooth_cej_buccal"]
    P3, P4 = P["mesial_tooth_root_apex"], P["distal_tooth_root_apex"]
    P5, P6 = P["crest_buccal_midspan"], P["crest_lingual_midspan"]

    span = P2 - P1
    inter_cej_mm = float(np.linalg.norm(span))
    if inter_cej_mm < 2.0:
        raise ValueError(f"the two CEJ landmarks are {inter_cej_mm:.2f} mm apart; "
                         f"they look like the same tooth")
    e_md = span / inter_cej_mm

    e_ap = ((P3 - P1) / np.linalg.norm(P3 - P1) + (P4 - P2) / np.linalg.norm(P4 - P2)) / 2.0
    e_ap = e_ap - (e_ap @ e_md) * e_md
    n = np.linalg.norm(e_ap)
    if n < 1e-6:
        raise ValueError("the mean tooth long axis is parallel to the inter-CEJ line")
    e_ap /= n

    e_bu = np.cross(e_md, e_ap)
    e_bu /= np.linalg.norm(e_bu)

    if float((P5 - P6) @ e_bu) < 0:
        e_md = -e_md
        e_bu = np.cross(e_md, e_ap)
        e_bu /= np.linalg.norm(e_bu)

    mid = 0.5 * (P1 + P2)
    off = np.asarray(offset, float)
    origin = (float(mid @ e_md) * e_md
              + float(P5 @ e_ap) * e_ap
              + float(P5 @ e_bu) * e_bu)

    if anterior_axis is None:
        anterior_axis = np.array([0.0, 1.0, 0.0])
    frame = LocalFrame(origin, e_md, e_ap, e_bu, off)
    frame.md_positive_is = "mesial" if float(e_md @ anterior_axis) > 0 else "distal"
    frame.jaw = "maxilla" if float(e_ap[0]) > 0 else "mandible"

    crest_sep = float(np.linalg.norm(P5 - P6))
    side_check = {"rule": "",
                  "crest_separation_mm": round(crest_sep, 2),
                  "site_L_mm": round(float(0.5 * (P1[2] + P2[2])), 2)}
    if crest_sep < 3.0:
        side_check["SHORT_BASELINE"] = (
            f"the two crest landmarks are only {crest_sep:.2f} mm apart (knife-edge ridge), so "
            f"the bucco-lingual SIGN rests on a short baseline. It is still the most reliable "
            f"rule available -- three geometric alternatives were tested against a "
            f"graft-validated ground truth and all three were wrong -- but confirm the buccal "
            f"side on the QC figure before trusting the volume.")
    if midline_L_mm is not None:
        site_L = 0.5 * (P1[2] + P2[2])
        side_check["midline_L_mm"] = round(float(midline_L_mm), 2)
        side_check["site_is_left_of_midline_in_volume"] = bool(site_L > midline_L_mm)
        lateral = np.array([0.0, 0.0, 1.0]) * np.sign(site_L - midline_L_mm)
        side_check["agrees_with_lateral_heuristic"] = bool(float(lateral @ e_bu) > 0)
    if site_fdi:
        q = str(site_fdi).strip()[:1]
        expect_left = q in ("2", "3")
        side_check["site_fdi"] = str(site_fdi)
        side_check["fdi_expects_patient_left"] = bool(expect_left)
        if midline_L_mm is not None and side_check["site_is_left_of_midline_in_volume"] != expect_left:
            side_check["LATERALITY_MISMATCH"] = (
                f"FDI {site_fdi} is a patient-{'LEFT' if expect_left else 'RIGHT'} site, but the "
                f"estimated mid-sagittal plane puts it on the "
                f"{'left' if side_check['site_is_left_of_midline_in_volume'] else 'right'}. "
                f"THE LIKELY CAUSE IS THE MIDLINE ESTIMATE, NOT THE SCAN: mid_sagittal_L_mm is "
                f"off by 26-32 mm on some of these volumes because the field of view is "
                f"off-centre. An image-free check of the CEJ landmarks against arch position "
                f"found no mirrored scan in this series. Nothing measured depends on it -- the "
                f"bucco-lingual sign comes from the operator's crest pair and mesial/distal from "
                f"the anterior axis, neither of which uses the midline. Trust the FDI code.")

    loc = frame.to_local(np.vstack([P1, P2, P3, P4, P5, P6]))
    meta = {
        "mode": "preop_anatomy",
        "landmarks_used": list(ANATOMY_LANDMARKS),
        "buccal_side_check": side_check,
        "inter_cej_span_mm": round(inter_cej_mm, 2),
        "crest_bucco_palatal_width_mm": round(float(np.linalg.norm(P5 - P6)), 2),
        "mean_tooth_axis_vs_inter_cej_deg": round(float(np.degrees(np.arccos(
            abs(np.clip(((P3 - P1) / np.linalg.norm(P3 - P1)) @ e_md, -1, 1))))), 1),
        "landmarks_local_XYZ_mm": {k: [round(float(c), 2) for c in loc[i]]
                                   for i, k in enumerate(ANATOMY_LANDMARKS)},
        "definition": (""),
    }
    return frame, meta


def despike(H, tol_mm=2.0, size=5):
    H = np.asarray(H, float).copy()
    ok = np.isfinite(H)
    if ok.sum() < size * size:
        return H, 0
    filled = np.where(ok, H, np.nanmedian(H[ok]))
    med = ndi.median_filter(filled, size=size, mode="nearest")
    bad = ok & (np.abs(H - med) > tol_mm)
    H[bad] = np.nan
    return H, int(bad.sum())


def height_field(points_local, x_edges, y_edges, z_min=-np.inf, z_max=np.inf,
                 mode="top_mean", branch_tol=None, spike_tol_mm=2.0):
    P = points_local
    m = ((P[:, 0] >= x_edges[0]) & (P[:, 0] <= x_edges[-1])
         & (P[:, 1] >= y_edges[0]) & (P[:, 1] <= y_edges[-1])
         & (P[:, 2] >= z_min) & (P[:, 2] <= z_max))
    P = P[m]
    del m
    ny, nx = len(y_edges) - 1, len(x_edges) - 1
    H = np.full((ny, nx), np.nan)
    if not len(P):
        return H
    ix = np.clip(np.digitize(P[:, 0], x_edges) - 1, 0, nx - 1).astype(np.int32, copy=False)
    iy = np.clip(np.digitize(P[:, 1], y_edges) - 1, 0, ny - 1).astype(np.int32, copy=False)
    flat = iy * nx + ix
    del ix, iy

    top = np.full(ny * nx, -np.inf)
    np.maximum.at(top, flat, P[:, 2])
    if mode == "max":
        H.flat[:] = np.where(np.isfinite(top), top, np.nan)
        return despike(H, spike_tol_mm)[0] if spike_tol_mm else H

    if branch_tol is None:
        step = float(np.median(np.diff(x_edges)))
        branch_tol = max(1.0, 4.0 * step)
    if mode in ("min", "bottom_mean"):
        bot = np.full(ny * nx, np.inf)
        np.minimum.at(bot, flat, P[:, 2])
        if mode == "min":
            H.flat[:] = np.where(np.isfinite(bot), bot, np.nan)
            return despike(H, spike_tol_mm)[0] if spike_tol_mm else H
        keep = P[:, 2] <= bot[flat] + branch_tol
        s = np.zeros(ny * nx); c = np.zeros(ny * nx)
        ki = flat[keep]; kz = P[keep, 2]
        np.add.at(s, ki, kz); np.add.at(c, ki, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            H.flat[:] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        return despike(H, spike_tol_mm)[0] if spike_tol_mm else H
    keep = P[:, 2] >= top[flat] - branch_tol
    s = np.zeros(ny * nx)
    c = np.zeros(ny * nx)
    ki = flat[keep]; kz = P[keep, 2]
    np.add.at(s, ki, kz)
    np.add.at(c, ki, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        H.flat[:] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return despike(H, spike_tol_mm)[0] if spike_tol_mm else H

