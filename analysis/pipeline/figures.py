from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

from . import geom

GAIN_SCALE = 8.0
DPI = 600
GREY = dict(cmap="gray", vmin=-400, vmax=2200)


def bl_extent(u, v, apical_up):
    return [u[-1], u[0], v[0], v[-1]] if apical_up else [u[-1], u[0], v[-1], v[0]]


def md_axis_label(md_positive_is, middle):
    left, right = (("distal", "mesial") if md_positive_is == "mesial" else ("mesial", "distal"))
    return f"$\\leftarrow$ {left}      {middle}      {right} $\\rightarrow$"


def bl_image(img, apical_up):
    return img[::-1, ::-1] if apical_up else img[:, ::-1]


def use_fonts(font_dir="fonts"):
    for f in ("Arimo-Regular.ttf", "Arimo-Bold.ttf", "Arimo-Italic.ttf"):
        try:
            font_manager.fontManager.addfont(f"{font_dir}/{f}")
        except Exception:
            pass
    plt.rcParams.update({"font.family": "Arimo", "font.size": 9,
                         "axes.linewidth": 0.8, "figure.dpi": 110,
                         "savefig.bbox": "tight", "pdf.fonttype": 42})


def _save(fig, stem, pdf=True, dpi=DPI):
    fig.savefig(f"{stem}.png", dpi=dpi)
    if pdf:
        fig.savefig(f"{stem}.pdf")
    plt.close(fig)
    return f"{stem}.png"


def slice_local(vol, frame, voxel_mm, fixed_axis, fixed_value,
                u_axis, u_range, v_axis, v_range, step=0.06):
    u = np.arange(u_range[0], u_range[1] + 1e-9, step)
    v = np.arange(v_range[0], v_range[1] + 1e-9, step)
    U, V = np.meshgrid(u, v, indexing="xy")
    Q = np.empty(U.shape + (3,))
    Q[..., fixed_axis] = fixed_value
    Q[..., u_axis] = U
    Q[..., v_axis] = V
    P = frame.to_anatomical(Q.reshape(-1, 3)) / voxel_mm
    ok = np.all((P >= 0) & (P < np.array(vol.shape) - 1), axis=1)
    out = np.full(len(P), np.nan)
    out[ok] = ndi.map_coordinates(vol, P[ok].T, order=1, mode="nearest")
    return out.reshape(U.shape), u, v


def fig_cross_section(case, pre_vol, post_vol, frame, voxel_mm, roi, thickness, xe, ye,
                      Hpre, Hpost, xe_h, ye_h, background=0.0,
                      out_dir="exports", z_lo=6.0, z_hi=30.0, pad=3.0, metal=None):
    use_fonts()
    col = np.nanmean(thickness, axis=0)
    ix = int(np.nanargmax(np.nan_to_num(col, nan=-1e9)))
    x_at = 0.5 * (xe[ix] + xe[ix + 1])

    y0, y1 = roi["Y_mm"][0] - pad, roi["Y_mm"][1] + pad
    jx0 = int(np.clip(np.searchsorted(xe_h, x_at) - 1, 0, Hpre.shape[1] - 1))
    band = np.concatenate([np.asarray(Hpre[:, jx0], float), np.asarray(Hpost[:, jx0], float)])
    band = band[np.isfinite(band)]
    if band.size:
        view_lo, view_hi = float(band.min()) - 3.5, float(band.max()) + 4.5
    else:
        view_lo, view_hi = z_lo - 3.0, z_hi + 2.0
    imgs = {}
    for nm, vol in (("pre", pre_vol), ("post", post_vol)):
        imgs[nm], u, v = slice_local(vol, frame, voxel_mm, 0, x_at, 2, (view_lo, view_hi),
                                     1, (y0, y1))
    img_metal = None
    if metal is not None:
        img_metal, _, _ = slice_local(np.multiply(metal, np.float32(1000.0),
                                                  dtype=np.float32), frame, voxel_mm, 0,
                                      x_at, 2, (view_lo, view_hi), 1, (y0, y1))

    jx = int(np.clip(np.searchsorted(xe_h, x_at) - 1, 0, Hpre.shape[1] - 1))
    yc = 0.5 * (ye_h[:-1] + ye_h[1:])
    cpre = np.asarray(Hpre[:, jx], float)
    cpost = np.asarray(Hpost[:, jx], float)

    panel_w, panel_h = (view_hi - view_lo), (y1 - y0)
    fig_w = 12.6
    fig_h = fig_w / (3.0 * panel_w / panel_h) + 1.15
    fig, ax = plt.subplots(1, 3, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    up = frame.apical_is_up_on_screen
    ext = bl_extent(u, v, up)

    def show(a, img, letter, title):
        a.imshow(bl_image(img, up), extent=ext, origin="upper", aspect="equal", **GREY)
        a.set_title(f"({letter})  {title}", fontsize=11, pad=6, loc="left")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    show(ax[0], imgs["pre"], "a", "Pre-operative")
    show(ax[1], imgs["post"], "b", "Post-operative")
    show(ax[2], imgs["post"], "c", "Superimposed")

    for a in (ax[1], ax[2]):
        if img_metal is not None and np.nanmax(img_metal) > 500:
            a.contour(u[::-1], v, img_metal[:, ::-1], levels=[500], colors=["#c9a227"],
                      linewidths=1.6)
    for a_ in ax:
        a_.set_ylim(v[0], v[-1]) if up else a_.set_ylim(v[-1], v[0])

    a = ax[2]
    inband = (yc >= roi["Y_mm"][0] - 0.4) & (yc <= roi["Y_mm"][1] + 0.4)
    cpre = np.where(inband, cpre, np.nan)
    cpost = np.where(inband, cpost, np.nan)
    a.plot(cpre, yc, color="#2f6fc8", lw=2.2, label="bone surface before", zorder=4)
    a.plot(cpost, yc, color="#c8432f", lw=2.2, label="bone surface after", zorder=4)
    good = np.isfinite(cpre) & np.isfinite(cpost) & (cpost > cpre)
    if good.any():
        a.fill_betweenx(yc[good], cpre[good], cpost[good], color="#c8432f", alpha=0.25,
                        lw=0, label="measured envelope")
        gain = np.where(good, cpost - cpre - background, np.nan)
        k = int(np.nanargmax(gain))
        a.annotate("", xy=(cpre[k], yc[k]), xytext=(cpost[k], yc[k]),
                   arrowprops=dict(arrowstyle="|-|,widthA=0.4,widthB=0.4", color="k", lw=1.3,
                                   shrinkA=0, shrinkB=0), zorder=6)
        a.text(cpost[k] + 2.6, yc[k], f"{gain[k]:.1f} mm", fontsize=10.5, ha="center",
               va="center", zorder=7,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", lw=0.6, alpha=0.93))
    if img_metal is not None and np.nanmax(img_metal) > 500:
        a.plot([], [], color="#c9a227", lw=1.6, label="implant (subtracted)")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.005))

    for a_ in ax:
        for yy in roi["Y_mm"]:
            a_.axhline(yy, color="k", lw=1.0, ls=(0, (4, 3)))
    a0 = ax[0]
    y_bot, y_top = (v[0], v[-1]) if up else (v[-1], v[0])
    xb = u[-1] - 1.0
    yb = y_bot + 0.9 * np.sign(y_top - y_bot)
    a0.plot([xb, xb - 5], [yb, yb], color="w", lw=3.4, solid_capstyle="butt")
    a0.text(xb - 2.5, yb + 0.35 * np.sign(y_top - y_bot), "5 mm", ha="center",
            va="bottom" if up else "top", fontsize=8.5, color="w")
    ya = y_top - 1.3 * np.sign(y_top - y_bot)
    a0.annotate("", xy=(xb, ya), xytext=(xb - 4.2, ya),
                arrowprops=dict(arrowstyle="-|>", color="w", lw=1.3))
    a0.text(xb - 4.6, ya, "buccal", fontsize=8.5, ha="left", va="center", color="w")
    fig.suptitle(f"{case} — buccal bone before and after augmentation", fontsize=13, y=0.995)
    fig.text(0.5, 0.932, "section across the ridge at the plane of greatest gain; "
                         "black dashes bound the region of interest",
             ha="center", fontsize=9.5, color="#444444")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.845, bottom=0.085, wspace=0.025)
    return _save(fig, f"{out_dir}/{case}_fig1_cross_section")


def _masked(cmap_name, bad="#9e9e9e"):
    cm = plt.get_cmap(cmap_name).copy()
    cm.set_bad(bad)
    return cm


def fig_gain_map(case, thickness, xe, ye, roi, controls=None, out_dir="exports",
                 ctrl_maps=None, apical_up=True, md_positive_is="mesial"):
    use_fonts()
    x0 = roi["X_mm"][0] + roi["size_md_mm"] / 2.0
    y0 = roi["Y_mm"][0]
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    norm = TwoSlopeNorm(vmin=-GAIN_SCALE, vcenter=0.0, vmax=GAIN_SCALE)
    cmap = _masked("RdBu_r")
    th_m = np.ma.masked_invalid(thickness)
    im = ax.pcolormesh(xe - x0, ye - y0, th_m, cmap=cmap, norm=norm, shading="auto")
    for T, X, Y in (ctrl_maps or []):
        ax.pcolormesh(X - x0, Y - y0, np.ma.masked_invalid(T), cmap=cmap, norm=norm,
                      shading="auto")
    xc = 0.5 * (xe[:-1] + xe[1:]) - x0
    yc = 0.5 * (ye[:-1] + ye[1:]) - y0
    with np.errstate(invalid="ignore"):
        ax.contour(xc, yc, th_m, levels=[-6, -4, -2, 0, 2, 4, 6],
                   colors="#333333", linewidths=0.55, alpha=0.75)
    def lab_x(a, b):
        return a + 0.3

    ra, rb = roi["X_mm"][0] - x0, roi["X_mm"][1] - x0
    ax.add_patch(plt.Rectangle((ra, 0.0), roi["size_md_mm"], roi["size_ap_mm"],
                               fill=False, ec="#111111", lw=2.2))
    ax.text(lab_x(ra, rb), 0.75, "ROI", fontsize=10, weight="bold", ha="left")
    for nm, xr in (controls or {}).items():
        a, b = xr[0] - x0, xr[1] - x0
        ax.add_patch(plt.Rectangle((a, 0.0), b - a, roi["size_ap_mm"],
                                   fill=False, ec="#444444", lw=1.2, ls=(0, (5, 3))))
        ax.text(lab_x(a, b), 0.75, f"control ({nm})", fontsize=8, color="#333333", ha="left")
    if not apical_up:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    anch = (roi or {}).get("anchor", "implant")
    if anch == "preop_anatomy":
        ax.set_xlabel(md_axis_label(md_positive_is,
                                    "along the ridge, mm from the inter-CEJ midpoint"))
        ax.set_ylabel("mm apical to the\npre-operative buccal crest")
    else:
        ax.set_xlabel(md_axis_label(md_positive_is, "along the ridge, mm from the implant axis"))
        ax.set_ylabel("mm apical to\nthe implant platform")
    finite = int(np.isfinite(thickness).sum()); total = int(thickness.size)
    ax.text(0.995, 0.02, f"{100*finite/max(total,1):.1f} % of cells resolved",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#9e9e9e", edgecolor="none",
                             label="no bone surface resolved in the search band (before or after)")],
              loc="upper right", fontsize=7.5, frameon=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.015, extend="both")
    cb.set_label("bucco-lingual bone gain (mm)")
    ax.set_title(f"{case} — bone gain across the buccal surface", fontsize=11, pad=8)
    fig.tight_layout()
    return _save(fig, f"{out_dir}/{case}_fig2_gain_map")


def fig_profiles(case, prof, out_dir="exports", md_label="along the ridge",
                 apical_up=True, md_positive_is="mesial", roi=None):
    use_fonts()
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.6))
    d, dm, dx = prof["_depth"], prof["_depth_mean"], prof["_depth_max"]
    ax[0].fill_betweenx(d, 0, dm, color="#c8432f", alpha=0.18)
    ax[0].plot(dm, d, "-", lw=2.2, color="#c8432f", label="mean along the ridge")
    ax[0].plot(dx, d, "--", lw=1.2, color="#8a2a1c", alpha=0.85, label="maximum along the ridge")
    if not apical_up:
        ax[0].invert_yaxis()
    ax[0].axvline(0, color="k", lw=0.6)
    ax[0].set_xlabel("bone gain (mm)")
    ax[0].set_ylabel("mm apical to the pre-operative buccal crest"
                     if (roi or {}).get("anchor") == "preop_anatomy"
                     else "mm apical to the implant platform")
    ax[0].set_title(f"(a)  Gain with depth — peak {prof['peak']['mean_mm']:.2f} mm "
                    f"at {prof['peak']['depth_mm']:.1f} mm apical", fontsize=10.5, loc="left")
    a, am, ax_ = prof["_along"], prof["_along_mean"], prof["_along_max"]
    ax[1].fill_between(a, 0, am, color="#2f6fc8", alpha=0.18)
    ax[1].plot(a, am, "-", lw=2.2, color="#2f6fc8", label="mean over depth")
    ax[1].plot(a, ax_, "--", lw=1.2, color="#1c4a8a", alpha=0.85, label="maximum over depth")
    ax[1].axvline(0, color="k", lw=0.8, ls=":")
    ax[1].set_xlabel(md_axis_label(
        md_positive_is,
        "mm from the inter-CEJ midpoint" if (roi or {}).get("anchor") == "preop_anatomy"
        else "mm from the implant axis"))
    ax[1].set_ylabel("bone gain (mm)")
    ax[1].set_title("(b)  Gain along the ridge", fontsize=10.5, loc="left")
    for a_ in ax:
        a_.grid(alpha=0.28, lw=0.5)
        a_.legend(fontsize=8, frameon=False)
    fig.suptitle(f"{case} — horizontal buccal bone gain", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, f"{out_dir}/{case}_fig3_profiles")


def fig_3d(case, mesh_local, Hpre, Hpost, xe, ye, roi, out_dir="exports", implant_vf=None,
           background=0.0, window=(1900, 1450), jaw="maxilla", md_positive_is="mesial"):
    import pyvista as pv
    pv.OFF_SCREEN = True

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    lo = np.asarray(Hpre, float).T
    hi = np.asarray(Hpost, float).T
    thick = hi - lo - background
    ok = np.isfinite(lo) & np.isfinite(hi)
    inroi = ((xc[:, None] >= roi["X_mm"][0]) & (xc[:, None] <= roi["X_mm"][1])
             & (yc[None, :] >= roi["Y_mm"][0]) & (yc[None, :] <= roi["Y_mm"][1]))
    ok &= inroi

    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")
    fill = float(np.nanmedian(hi[np.isfinite(hi)])) if np.isfinite(hi).any() else 15.0
    Z0 = np.where(ok, lo, fill)
    Z1 = np.where(ok, hi, fill)
    X3 = np.stack([Xg, Xg], axis=-1)
    Y3 = np.stack([Yg, Yg], axis=-1)
    Z3 = np.stack([Z0, Z1], axis=-1)
    grid = pv.StructuredGrid(X3, Y3, Z3)
    grid.point_data["gain_mm"] = np.stack(
        [np.where(ok, np.clip(thick, 0.0, None), 0.0)] * 2, axis=-1).ravel(order="F")
    cell_ok = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    grid.cell_data["valid"] = cell_ok.ravel(order="F").astype(np.uint8)
    solid = grid.threshold(0.5, scalars="valid").extract_surface()

    V, F = mesh_local
    V = np.asarray(V, float)
    zc = float(np.nanmedian(lo[np.isfinite(lo)])) if np.isfinite(lo).any() else 15.0
    cx0, cx1 = roi["X_mm"][0] - 13.0, roi["X_mm"][1] + 13.0
    cy0, cy1 = roi["Y_mm"][0] - 5.0, roi["Y_mm"][1] + 2.0
    cz0, cz1 = zc - 7.0, zc + 10.0
    kv = ((V[:, 0] >= cx0) & (V[:, 0] <= cx1) & (V[:, 1] >= cy0) & (V[:, 1] <= cy1)
          & (V[:, 2] >= cz0) & (V[:, 2] <= cz1))
    kf = kv[F].all(axis=1)
    used = np.unique(F[kf])
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    Fc = remap[F[kf]]
    ctx = pv.PolyData(V[used], np.hstack([np.full((len(Fc), 1), 3, np.int64), Fc]).ravel())

    p = pv.Plotter(off_screen=True, window_size=list(window), lighting="three lights")
    p.set_background("white")
    if ctx.n_cells:
        p.add_mesh(ctx.smooth(n_iter=20, relaxation_factor=0.08, boundary_smoothing=False),
                   color="#d8d4cc", opacity=1.0, smooth_shading=True, show_scalar_bar=False,
                   ambient=0.26, diffuse=0.74, specular=0.05)
    if implant_vf is not None:
        Vi, Fi = implant_vf
        fi = np.hstack([np.full((len(Fi), 1), 3, np.int64), np.asarray(Fi, np.int64)]).ravel()
        p.add_mesh(pv.PolyData(np.asarray(Vi, float), fi), color="#b08d2a", smooth_shading=True,
                   show_scalar_bar=False, ambient=0.22, specular=0.35, opacity=1.0)
    p.add_mesh(solid.smooth(n_iter=8, relaxation_factor=0.04, boundary_smoothing=False),
               scalars="gain_mm", cmap="YlOrRd", clim=(0.0, GAIN_SCALE),
               smooth_shading=True, ambient=0.22, diffuse=0.80, specular=0.22, specular_power=24,
               scalar_bar_args=dict(title="graft thickness (mm)", n_labels=5, vertical=True,
                                    position_x=0.90, position_y=0.30, height=0.42, width=0.030,
                                    title_font_size=20, label_font_size=16, color="black",
                                    font_family="arial"))

    left, right = (("distal", "mesial") if md_positive_is == "mesial" else ("mesial", "distal"))
    p.add_text(left, position=(0.04, 0.04), viewport=True, font_size=14, color="#333333", font="arial")
    p.add_text(right, position=(0.74, 0.04), viewport=True, font_size=14, color="#333333", font="arial")
    p.add_text(f"{case} — the augmented bone, rendered as a solid",
               position=(0.03, 0.945), viewport=True, font_size=19, color="black", font="arial")

    ctr = np.array([0.5 * (cx0 + cx1), 0.5 * (cy0 + cy1), zc])
    half = 0.5 * max(cx1 - cx0, cy1 - cy0)
    d = np.array([0.45, -0.70, 0.90])
    d /= np.linalg.norm(d)
    up = (0.0, 1.0, 0.0) if jaw == "maxilla" else (0.0, -1.0, 0.0)
    p.enable_parallel_projection()
    p.camera_position = [tuple(ctr + d * 6.0 * half), tuple(ctr), up]
    p.camera.parallel_scale = 1.05 * half
    sx = cx0 + 1.5
    sy = cy1 - 1.5
    p.add_lines(np.array([[sx, sy, cz1], [sx + 5.0, sy, cz1]]), color="black", width=5)
    p.add_point_labels(np.array([[sx + 2.5, sy, cz1]]), ["5 mm"], font_size=18, bold=True,
                       text_color="black", shape=None, show_points=False,
                       always_visible=True, justification_horizontal="center")
    try:
        p.enable_ssao(radius=1.2, bias=0.02)
    except Exception:
        pass
    p.enable_anti_aliasing("ssaa")
    path = f"{out_dir}/{case}_fig4_3d.png"
    p.screenshot(path, scale=2)
    p.close()
    return path


def fig_qc(case, qc, pre_vol, post_vol, frame, voxel_mm, roi, out_dir="exports"):
    use_fonts()
    fig = plt.figure(figsize=(13.2, 8.4))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    img, u, v = slice_local(post_vol, frame, voxel_mm, 1, roi["Y_mm"][0] + 2.0,
                            2, (15.0 - 12, 15.0 + 14), 0, (roi["X_mm"][0] - 4, roi["X_mm"][1] + 4))
    ax.imshow(img[:, ::-1], extent=[u[-1], u[0], v[-1], v[0]], origin="upper",
              aspect="equal", **GREY)
    ax.annotate("", xy=(15.0 + 12, np.mean(v)), xytext=(15.0 + 2, np.mean(v)),
                arrowprops=dict(arrowstyle="-|>", color="#39b54a", lw=2.6))
    ax.text(15.0 + 7, np.mean(v) - 1.0, "BUCCAL", color="#158a2a", fontsize=9,
            va="bottom", ha="center", weight="bold")
    _ref = ("the pre-operative buccal crest" if roi.get("anchor") == "preop_anatomy"
            else "the implant platform")
    ax.set_title(f"(a) buccal direction\n(2 mm apical to {_ref})", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    img, u, v = slice_local(post_vol, frame, voxel_mm, 0, 15.0,
                            2, (15.0 - 12, 15.0 + 14), 1, (roi["Y_mm"][0] - 4, roi["Y_mm"][1] + 4))
    up = frame.apical_is_up_on_screen
    ax.imshow(bl_image(img, up), origin="upper", aspect="equal", **GREY,
              extent=bl_extent(u, v, up))
    imp = qc.get("implant", {})
    if imp.get("present"):
        xlab = u[-1] - 0.4
        ax.axhline(roi["Y_mm"][0], color="#c9a227", lw=1.4)
        ax.text(xlab, roi["Y_mm"][0] + (0.5 if up else -0.5), "platform", color="#c9a227",
                fontsize=8.5, ha="left", va="bottom" if up else "top")
        ax.axhline(roi["Y_mm"][0] + imp.get("height_mm", 0), color="#c9a227", lw=1.0, ls=":")
        ax.text(xlab, roi["Y_mm"][0] + imp.get("height_mm", 0) - (0.5 if up else -0.5), "apex",
                color="#c9a227", fontsize=8.5, ha="left", va="top" if up else "bottom")
        t = f"implant {imp.get('height_mm', 0):.1f} mm, {100 * imp.get('new_in_post_frac', 0):.0f} % new"
    elif imp.get("declared_absent"):
        t = "no implant in this case (declared)"
    else:
        t = "no implant detected"
    ax.set_title(f"(b) {t}", fontsize=9.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    acc = qc.get("accuracy", {})
    names = [k for k in ("control distal", "control mesial", "rigid shift", "dilation")
             if acc.get(k) is not None]
    vals = [acc[k] for k in names]
    if names:
        cols = ["#8fbcd4", "#8fbcd4", "#c8432f", "#e0a458"][:len(names)]
        ax.bar(range(len(names)), vals, color=cols, width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        for i, val in enumerate(vals):
            ax.text(i, val + 0.02 * (1 if val >= 0 else -1), f"{val:+.2f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=8.5)
    for sgn in (1, -1):
        ax.axhline(sgn * voxel_mm, color="#777777", ls="--", lw=0.9)
    ax.text(0.01, voxel_mm + 0.01, "1 voxel", color="#777777", fontsize=7.5,
            transform=ax.get_yaxis_transform())
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("surface displacement (mm)")
    ax.set_title("(c) superimposition accuracy\nmeasured on bone that cannot have changed",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="y", lw=0.5)

    ax = fig.add_subplot(gs[1, 0])
    gsens = qc.get("grid_sensitivity", {})
    if gsens:
        ks = sorted(gsens, key=float)
        ax.plot([float(k) for k in ks], [gsens[k] for k in ks], "-o", color="#2f6fc8")
        ax.set_xlabel("grid step (mm)")
        ax.set_ylabel("gain volume (mm$^3$)")
        ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("(d) grid-step sensitivity", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 1])
    tsens = qc.get("threshold_sensitivity", {})
    if tsens:
        ks = sorted(tsens, key=float)
        ax.plot([float(k) for k in ks], [tsens[k] for k in ks], "-o", color="#c8432f")
        ax.set_xlabel("segmentation threshold (grey value)")
        ax.set_ylabel("gain volume (mm$^3$)")
        ax.grid(alpha=0.25, lw=0.5)
    ax.set_title("(e) threshold sensitivity", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lines = qc.get("summary_lines", [])
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.6, family="monospace")
    ax.set_title("(f) reported values", fontsize=9.5, loc="left")

    fig.suptitle(f"{case} — quality control", fontsize=12)
    return _save(fig, f"{out_dir}/{case}_fig5_qc", pdf=False, dpi=300)

