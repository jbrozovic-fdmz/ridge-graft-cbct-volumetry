import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

from . import geom
from .arch import ArchCurve

DPI = 600
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
         "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}
GREY = dict(cmap="gray", interpolation="bilinear")
PRE_C, POST_C = "#56B4E9", "#D55E00"
SECTION_HALF_MM = 16.0
SECTION_ABOVE_MM, SECTION_BELOW_MM = 20.0, 8.0
DEV_VMAX = 6.0
DEV_VMIN = -1.5


def use_fonts(font_dir="fonts"):
    d = Path(font_dir)
    if d.is_dir():
        for f in d.glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = "Arimo"
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.7,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.5, "figure.dpi": 110,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
                         "axes.spines.top": False, "axes.spines.right": False})


def panel_tag(ax, letter, dx=-0.055, dy=1.055):
    if hasattr(ax, "zaxis"):
        ax.text2D(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=11,
                  fontweight="bold", va="top", ha="left")
    else:
        ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left")


def dev_norm():
    return TwoSlopeNorm(vmin=DEV_VMIN, vcenter=0.0, vmax=DEV_VMAX)


def dev_cmap():
    cm = plt.get_cmap("RdYlBu_r").copy()
    cm.set_bad("#bdbdbd")
    return cm


class Case:

    def __init__(self, case, reader="A"):
        self.case = case
        self.reader = reader
        rp = (f"analysis/{case}/results.json" if reader == "A"
              else f"analysis/{case}/reader{reader}/results.json")
        self.res = json.loads(Path(rp).read_text())
        self.frame = geom.LocalFrame.from_dict(self.res["frame"])
        self.roi = self.res["roi"]
        self.prof = self.res["new_bone_md_profile"]
        self.arch_json = json.loads(Path(f"cases/{case}_arch.json").read_text())
        self._arch = None
        self._vols = {}

    @property
    def arch(self):
        if self._arch is None:
            a = ArchCurve(np.array(self.arch_json["arch_points_AL_mm"], float))
            a.orient_normal_outward(a.points.mean(axis=0))
            eb = np.asarray(self.res["frame"]["rows_e_md_e_apical_e_buccal"][2], float)
            st = self._station_raw(a)
            dot = float(a.normal[st, 0] * eb[1] + a.normal[st, 1] * eb[2])
            self.arch_normal_dot_e_buccal = round(dot, 3)
            if dot < 0:
                a.normal *= -1.0
            if abs(dot) < 0.5:
                raise SystemExit(
                    f"ABORT ({self.case}): the arch normal and the frame's buccal axis are "
                    f"{np.degrees(np.arccos(min(1, abs(dot)))):.0f} deg apart. One of the two is "
                    f"wrong and a section drawn from either would not be the plane the volume "
                    f"was measured in. Re-check cases/{self.case}_arch.json against the crest "
                    f"landmarks before drawing anything.")
            self._arch = a
        return self._arch

    def _station_raw(self, a):
        y = float(np.mean(self.roi["Y_mm"]))
        w = self.frame.to_anatomical(np.array([[self.peak_local_x(), y,
                                                self.frame.offset[2]]]))[0]
        return a.nearest_station(w[1], w[2])

    def volume(self, which):
        if which not in self._vols:
            path = (f"analysis/{self.case}/pre_cal.nii.gz" if which == "pre"
                    else f"cbct_blinded/{self.case}/post_reg.nii.gz")
            self._vols[which] = geom.load_volume(path)
        return self._vols[which]

    def drop_volumes(self):
        self._vols.clear()

    def gain_map(self):
        d = np.genfromtxt(f"analysis/{self.case}/gain_map.csv", delimiter=",", names=True)
        xs = np.unique(d["along_ridge_mm"]); ys = np.unique(d["apical_mm"])
        G = np.full((len(ys), len(xs)), np.nan)
        G[np.searchsorted(ys, d["apical_mm"]), np.searchsorted(xs, d["along_ridge_mm"])] = \
            d["gain_mm"]
        return xs, ys, G

    def peak_local_x(self):
        xs, _, G = self.gain_map()
        with np.errstate(invalid="ignore"):
            col = np.nanmean(np.where(np.isfinite(G), G, np.nan), axis=0)
        return float(xs[int(np.nanargmax(col))]) + self.frame.offset[0]

    def station_at_local_x(self, x_local):
        y = float(np.mean(self.roi["Y_mm"]))
        w = self.frame.to_anatomical(np.array([[x_local, y, self.frame.offset[2]]]))[0]
        return self.arch.nearest_station(w[1], w[2])

    def crest_S_mm(self):
        c = self.frame.to_anatomical(np.array([[self.frame.offset[0], self.roi["Y_mm"][0],
                                                self.frame.offset[2]]]))[0]
        return float(c[0])

    def section(self, which, station):
        vol, vx = self.volume(which)
        cs = self.crest_S_mm()
        s_lo = max(0.0, cs - SECTION_BELOW_MM)
        s_hi = min((vol.shape[0] - 1) * vx, cs + SECTION_ABOVE_MM)
        return self.arch.section(vol, vx, station, half_width_mm=SECTION_HALF_MM,
                                 s_range=(s_lo, s_hi))


def _buccal_trace(field, u, s_ax, min_run=2):
    out = np.full(len(s_ax), np.nan)
    bone = field > geom.ISO_LEVEL
    uu = u
    for r in range(bone.shape[0]):
        row = bone[r]
        run = 0
        for j in range(len(row) - 1, -1, -1):
            if row[j]:
                run += 1
                if run >= min_run:
                    out[r] = uu[min(j + min_run - 1, len(uu) - 1)]
                    break
            else:
                run = 0
    return out


def panel_section(ax, C, station=None, show_roi=True, title=None):
    st = C.station_at_local_x(C.peak_local_x()) if station is None else station
    post_raw, ext, u, s_ax = C.section("post", st)
    pre_raw, _, _, _ = C.section("pre", st)
    pix = (ext[1] - ext[0]) / (post_raw.shape[1] - 1)
    sig = geom.SMOOTH_SIGMA * C.volume("pre")[1] / pix
    def _field(a):
        f = np.where(a > geom.METAL_LEVEL, -1000.0, a).astype(np.float32)
        return ndi.gaussian_filter(f, sig)
    pre, post = _field(pre_raw), _field(post_raw)

    ax.imshow(post_raw, origin="lower", extent=ext, aspect="equal", vmin=-400, vmax=1400,
              **GREY)

    cs = C.crest_S_mm()
    dep = C.roi["Y_mm"][1] - C.roi["Y_mm"][0]
    sgn = 1.0 if C.frame.jaw == "maxilla" else -1.0
    band = sorted((cs, cs + sgn * dep))
    pre_b = _buccal_trace(pre, u, s_ax)
    post_b = _buccal_trace(post, u, s_ax)
    shown = (s_ax >= band[0] - 3.0) & (s_ax <= band[1] + 3.0)
    meas = (s_ax >= band[0]) & (s_ax <= band[1]) & np.isfinite(pre_b) & np.isfinite(post_b)
    ax.fill_betweenx(s_ax[meas], pre_b[meas], post_b[meas],
                     where=(post_b[meas] > pre_b[meas]), color=POST_C, alpha=0.42,
                     lw=0, interpolate=False, label="bone gained (this section)")
    ax.plot(pre_b[shown], s_ax[shown], "-", color=PRE_C, lw=1.7,
            label="pre-operative buccal surface")
    ax.plot(post_b[shown], s_ax[shown], "-", color=POST_C, lw=1.4,
            label="post-operative buccal surface")

    if show_roi:
        ax.axhline(cs, color=OKABE["green"], lw=1.0, ls="--")
        ax.text(ext[1] - 0.5, cs, " crest", color=OKABE["green"], fontsize=7,
                ha="right", va="bottom" if sgn > 0 else "top")
        ax.axhline(cs + sgn * dep, color=OKABE["green"], lw=1.0, ls=":")
        ax.text(ext[1] - 0.5, cs + sgn * dep, f" {dep:.0f} mm apical", color=OKABE["green"],
                fontsize=7, ha="right", va="top" if sgn > 0 else "bottom")
    ax.annotate("", xy=(ext[1] - 1.0, ext[2] + 1.4), xytext=(ext[1] - 5.0, ext[2] + 1.4),
                arrowprops=dict(arrowstyle="-|>", color="w", lw=1.6))
    ax.text(ext[1] - 3.0, ext[2] + 1.9, "buccal", color="w", fontsize=7.5,
            ha="center", va="bottom", weight="bold")
    ax.plot([ext[0] + 1.0, ext[0] + 6.0], [ext[2] + 1.2] * 2, "-", color="w", lw=2.4)
    ax.text(ext[0] + 3.5, ext[2] + 1.7, "5 mm", color="w", fontsize=7.5, ha="center",
            va="bottom")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title or f"section {st} of {len(C.arch.points) - 1}, "
                          f"perpendicular to the arch", fontsize=8.5)
    return st


def panel_deviation(ax, C, cbar=True, controls=True):
    xs, ys, G = C.gain_map()
    im = ax.pcolormesh(xs, ys, np.ma.masked_invalid(G), cmap=dev_cmap(), norm=dev_norm(),
                       shading="nearest", rasterized=True)
    with np.errstate(invalid="ignore"):
        cs = ax.contour(xs, ys, np.ma.masked_invalid(G), levels=[0, 1, 2, 3, 4, 5],
                        colors="#222222", linewidths=0.45, alpha=0.65)
    ax.clabel(cs, fmt="%.0f", fontsize=6, inline=True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel(f"along the ridge, mm from the inter-CEJ midpoint  "
                  f"(+ = {C.frame.md_positive_is})")
    ax.set_ylabel("mm apical to the\npre-operative buccal crest")
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend="both")
        cb.set_label("bone gain (mm)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return im


def panel_profile(ax, C):
    off = C.frame.offset[0]
    x = np.asarray(C.prof["x_mm"], float)
    v = np.asarray(C.prof["new_bone_mm3_per_slab"], float)
    ax.fill_between(x, 0, v, color=OKABE["blue"], alpha=0.20, lw=0)
    ax.plot(x, v, "-", lw=1.6, color=OKABE["blue"])
    x0, x1 = C.roi["X_mm"][0] - off, C.roi["X_mm"][1] - off
    top = float(np.nanmax(v)) * 1.28 if np.isfinite(v).any() else 1.0
    ax.set_ylim(0, top)
    lab_y = top * 0.97
    ax.axvspan(x0, x1, color=OKABE["green"], alpha=0.13, lw=0)
    ax.text((x0 + x1) / 2, lab_y, "measured\nwindow", ha="center",
            va="top", fontsize=7.2, color="#14664c")
    used, lo, hi = set(), x0, x1
    for nm, (a, b) in (C.res.get("control_windows") or {}).items():
        ok = (C.res.get("controls", {}).get("windows", {}).get(nm, {}).get("valid", True))
        ax.axvspan(a - off, b - off, color="#666666" if ok else OKABE["red"],
                   alpha=0.10 if ok else 0.16, lw=0)
        ax.text((a + b) / 2 - off, lab_y,
                f"control\n({nm})" + ("" if ok else "\nexcluded"), ha="center", va="top",
                fontsize=7, color="#444444" if ok else "#8a3b12")
        used.add(nm)
        lo = min(lo, a - off); hi = max(hi, b - off)
    ex = C.res["graft_extent_SECONDARY"]
    ax.axhline(ex["threshold_mm3_per_mm"], color="#999999", lw=0.8, ls="--")
    ax.text(hi + 4, ex["threshold_mm3_per_mm"], " placement threshold", ha="left",
            va="bottom", fontsize=6.8, color="#777777")
    ax.set_xlabel(f"along the ridge, mm from the inter-CEJ midpoint  "
                  f"(+ = {C.frame.md_positive_is})")
    ax.set_ylabel("new bone (mm³ per 1 mm slab)")
    ax.set_xlim(max(x[0], lo - 8), min(x[-1], hi + 16))
    ax.grid(alpha=0.25, lw=0.5)


def panel_surface3d(ax, C, elev=26, azim=-58):
    xs, ys, G = C.gain_map()
    X, Y = np.meshgrid(xs, ys)
    Hpost = np.ma.masked_invalid(G)
    ax.plot_surface(X, Y, np.zeros_like(X), color="#9e9e9e", alpha=0.32, linewidth=0,
                    antialiased=True, shade=True, rstride=2, cstride=2)
    ax.plot_surface(X, Y, Hpost.filled(np.nan), facecolors=dev_cmap()(dev_norm()(G)),
                    linewidth=0, antialiased=True, rstride=1, cstride=1, shade=False)
    ax.set_xlabel("along ridge (mm)", labelpad=-4, fontsize=7)
    ax.set_ylabel("apical (mm)", labelpad=-4, fontsize=7)
    ax.set_zlabel("gain (mm)", labelpad=-4, fontsize=7)
    ax.tick_params(labelsize=6, pad=-2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((2.2, 1.9, 1.15), zoom=1.28)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03)


def save(fig, stem, out_dir, pdf=True):
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    png = d / f"{stem}.png"
    fig.savefig(png, dpi=DPI)
    if pdf:
        fig.savefig(d / f"{stem}.pdf")
    plt.close(fig)
    return str(png)

