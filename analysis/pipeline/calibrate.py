from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy import signal

AIR_HU = -1000.0
SOFT_HU = 60.0

PADDING_BELOW = -1200.0
PADDING_MIN_FRACTION = 0.01
MIN_SEPARATION = 400.0
MAX_SEPARATION = 2500.0


def _sample(vol, step=2):
    return vol[::step, ::step, ::step].ravel()


def padding_value(vol):
    v = _sample(vol)
    lo = float(v.min())
    if lo >= PADDING_BELOW:
        return None
    if float((v == lo).mean()) < PADDING_MIN_FRACTION:
        return None
    return lo


def grey_references(vol):
    v = _sample(vol)
    pad = padding_value(vol)
    if pad is not None:
        v = v[v > pad + 1.0]
    lo, hi = np.percentile(v, [0.1, 99.0])
    if not np.isfinite([lo, hi]).all() or hi - lo < MIN_SEPARATION:
        raise SystemExit("ABORT: grey-value calibration failed — the intensity range is degenerate.")
    h, edges = np.histogram(v, bins=500, range=(float(lo), float(hi)))
    centres = 0.5 * (edges[:-1] + edges[1:])
    hs = ndi.gaussian_filter1d(h.astype(float), 3.0)
    pk, _ = signal.find_peaks(np.concatenate([[0.0], hs, [0.0]]), prominence=hs.max() * 0.03)
    peaks = centres[np.clip(pk - 1, 0, len(centres) - 1)]
    if len(peaks) < 2:
        raise SystemExit(
            f"ABORT: grey-value calibration failed — expected an air peak and a soft-tissue peak, "
            f"found {len(peaks)} ({np.round(peaks).tolist()}). Inspect the scan; do not report a "
            f"volume from it.")
    air = float(peaks[0])
    later = peaks[peaks >= air + MIN_SEPARATION]
    if not len(later):
        raise SystemExit(
            f"ABORT: grey-value calibration failed — no soft-tissue peak at least "
            f"{MIN_SEPARATION:.0f} above the air peak {air:.0f} (peaks {np.round(peaks).tolist()}).")
    soft = float(later[0])
    if not (MIN_SEPARATION <= soft - air <= MAX_SEPARATION):
        raise SystemExit(
            f"ABORT: implausible air->soft separation {soft - air:.0f} (air {air:.0f}, "
            f"soft {soft:.0f}).")
    return {"air": round(air, 1), "soft": round(soft, 1), "span": round(soft - air, 1),
            "padding_value": None if pad is None else round(pad, 1),
            "n_peaks_found": int(len(peaks)),
            "peaks": np.round(peaks[:6], 1).tolist()}


def calibrate_array(vol, ref):
    scale = (SOFT_HU - AIR_HU) / (ref["soft"] - ref["air"])
    out = vol.astype(np.float32)
    if out is vol:
        out = out.copy()
    if ref["padding_value"] is not None:
        cut = 0.5 * (ref["padding_value"] + ref["air"])
        np.copyto(out, np.float32(ref["air"]), where=out < cut)
    np.subtract(out, ref["air"], out=out)
    np.multiply(out, scale, out=out)
    np.add(out, AIR_HU, out=out)
    if ref["padding_value"] is not None:
        np.maximum(out, AIR_HU - 500.0, out=out)
    return out


def calibrate_image(in_path, out_path):
    img = sitk.ReadImage(str(in_path))
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    ref = grey_references(a)
    b = calibrate_array(a, ref)
    del a
    out = sitk.GetImageFromArray(b)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(out_path))
    scale = (SOFT_HU - AIR_HU) / (ref["soft"] - ref["air"])
    return {"source": str(in_path), "calibrated": str(out_path),
            "air_grey": ref["air"], "soft_tissue_grey": ref["soft"], "span_grey": ref["span"],
            "scale": round(scale, 4),
            "offset": round(AIR_HU - ref["air"] * scale, 1),
            "padding_value": ref["padding_value"],
            "histogram_peaks": ref["peaks"],
            "targets": {"air": AIR_HU, "soft_tissue": SOFT_HU},
            "native_equivalent_of_iso450": round((450.0 - AIR_HU) / scale + ref["air"], 1),
            "note": ""}

