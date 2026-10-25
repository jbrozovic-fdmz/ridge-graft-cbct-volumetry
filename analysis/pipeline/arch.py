from __future__ import annotations

import numpy as np
from scipy import interpolate, ndimage as ndi


class ArchCurve:

    def __init__(self, points_AL, step_mm=1.0, smooth=None):
        p = np.asarray(points_AL, float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("arch points must be an (n, 2) array of (A, L) millimetres")
        keep = [0]
        for i in range(1, len(p)):
            if np.hypot(*(p[i] - p[keep[-1]])) > 1e-3:
                keep.append(i)
        self.dropped = len(p) - len(keep)
        p = p[keep]
        if len(p) < 4:
            raise ValueError(f"the arch needs at least 4 distinct points; got {len(p)} "
                             f"after dropping {self.dropped} coincident one(s)")
        k = min(3, len(p) - 1)
        s = float(len(p)) if smooth is None else float(smooth)
        try:
            tck, _ = interpolate.splprep([p[:, 0], p[:, 1]], s=s, k=k)
        except Exception as e:
            raise ValueError(
                f"could not fit a curve through these {len(p)} points ({type(e).__name__}: {e}). "
                f"Points must be placed in order along the arch and must not double back on "
                f"themselves.") from e
        u = np.linspace(0.0, 1.0, 4000)
        a, l = interpolate.splev(u, tck)
        d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(a), np.diff(l)))])
        self.length_mm = float(d[-1])
        n = max(int(round(self.length_mm / step_mm)) + 1, 2)
        target = np.linspace(0.0, self.length_mm, n)
        uu = np.interp(target, d, u)
        aa, ll = interpolate.splev(uu, tck)
        da, dl = interpolate.splev(uu, tck, der=1)
        t = np.stack([da, dl], 1)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        self.points = np.stack([aa, ll], 1)
        self.tangent = t
        self.normal = nrm
        self.station_mm = target
        self.step_mm = step_mm
        self._tck = tck
        self._input = p

    def orient_normal_outward(self, centroid_AL):
        c = np.asarray(centroid_AL, float)
        away = self.points - c[None, :]
        flip = np.sum(self.normal * away, axis=1) < 0
        self.normal[flip] *= -1.0
        return self

    def nearest_station(self, A, L):
        d = np.hypot(self.points[:, 0] - A, self.points[:, 1] - L)
        return int(np.argmin(d))

    def to_world(self, station, u_mm, s_mm):
        p = self.points[station]
        nrm = self.normal[station]
        return np.array([s_mm, p[0] + u_mm * nrm[0], p[1] + u_mm * nrm[1]], float)

    def section(self, vol, voxel_mm, station, half_width_mm=18.0, s_range=None,
                pixel_mm=None):
        pix = float(pixel_mm or voxel_mm)
        Z, A, L = vol.shape
        if s_range is None:
            s_range = (0.0, (Z - 1) * voxel_mm)
        u = np.arange(-half_width_mm, half_width_mm + pix, pix)
        s = np.arange(s_range[0], s_range[1] + pix, pix)
        p = self.points[station]
        nrm = self.normal[station]
        UU, SS = np.meshgrid(u, s, indexing="xy")
        aa = p[0] + UU * nrm[0]
        ll = p[1] + UU * nrm[1]
        idx = np.stack([SS.ravel() / voxel_mm, aa.ravel() / voxel_mm, ll.ravel() / voxel_mm])
        img = ndi.map_coordinates(vol, idx, order=1, mode="constant", cval=-1000.0)
        return img.reshape(SS.shape), (u[0], u[-1], s[0], s[-1]), u, s

    def panoramic(self, vol, voxel_mm, half_width_mm=12.0, pixel_mm=None, mode="max"):
        pix = float(pixel_mm or voxel_mm)
        Z = vol.shape[0]
        u = np.arange(-half_width_mm, half_width_mm + pix, pix)
        s = np.arange(0.0, (Z - 1) * voxel_mm, pix)
        out = np.empty((len(s), len(self.points)), np.float32)
        for i in range(len(self.points)):
            p, nrm = self.points[i], self.normal[i]
            UU, SS = np.meshgrid(u, s, indexing="xy")
            aa = p[0] + UU * nrm[0]
            ll = p[1] + UU * nrm[1]
            idx = np.stack([SS.ravel() / voxel_mm, aa.ravel() / voxel_mm, ll.ravel() / voxel_mm])
            sl = ndi.map_coordinates(vol, idx, order=1, mode="constant",
                                     cval=-1000.0).reshape(SS.shape)
            out[:, i] = sl.max(axis=1) if mode == "max" else sl.mean(axis=1)
        return out, s

