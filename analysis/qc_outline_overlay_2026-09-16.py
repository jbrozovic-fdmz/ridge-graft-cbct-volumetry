#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pipeline import geom
from pipeline.arch import ArchCurve

HALF_W, ABOVE, BELOW = 20.0, 12.0, 32.0
GREY = (-400.0, 1400.0)
FIG_STATION = {"S1": 26, "S3": 21, "S2": 21, "S4": 23}
ROW_ORDER = ["S1", "S3", "S2", "S4"]


def case_geometry(case):
    res = json.loads(Path(f"analysis/{case}/results.json").read_text())
    fr = geom.LocalFrame.from_dict(res["frame"])
    aj = json.loads(Path(f"cases/{case}_arch.json").read_text())
    arch = ArchCurve(np.array(aj["arch_points_AL_mm"], float))
    arch.orient_normal_outward(arch.points.mean(axis=0))
    eb = np.array(res["frame"]["rows_e_md_e_apical_e_buccal"][2], float)
    y = float(np.mean(res["roi"]["Y_mm"]))
    w = fr.to_anatomical(np.array([[fr.offset[0], y, fr.offset[2]]]))[0]
    st0 = arch.nearest_station(w[1], w[2])
    if float(arch.normal[st0, 0] * eb[1] + arch.normal[st0, 1] * eb[2]) < 0:
        arch.normal *= -1.0
    return res, fr, arch


def crest_S(res, fr):
    c = fr.to_anatomical(np.array([[fr.offset[0], res["roi"]["Y_mm"][0], fr.offset[2]]]))[0]
    return float(c[0])


def section(vol_vx, arch, station, cs):
    vol, vx = vol_vx
    a, b = sorted((cs - ABOVE, cs + BELOW))
    s_lo, s_hi = max(0.0, a), min((vol.shape[0] - 1) * vx, b)
    return arch.section(vol, vx, station, half_width_mm=HALF_W, s_range=(s_lo, s_hi))


def us_to_local(us, arch, station, fr):
    V = np.asarray(us, float)
    if len(V) == 0:
        return np.zeros((0, 3))
    A0, L0 = arch.points[station]
    nA, nL = arch.normal[station]
    P = np.column_stack([V[:, 1], A0 + V[:, 0] * nA, L0 + V[:, 0] * nL])
    return fr.to_local(P)


def plane_in_section(arch, station, fr, y_value, u_span, n=64):
    us = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Q = us_to_local(us, arch, station, fr)
    y0 = Q[0, 1]
    dy_du = Q[1, 1] - y0
    dy_ds = Q[2, 1] - y0
    if abs(dy_ds) < 1e-9:
        return None
    u = np.linspace(u_span[0], u_span[1], n)
    s = (y_value - y0 - dy_du * u) / dy_ds
    return u, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--case", action="append")
    ap.add_argument("--station", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = args.case if args.case else ROW_ORDER

    report = {}
    fig, axes = plt.subplots(len(cases), 2, figsize=(9.0, 4.6 * len(cases)))
    axes = np.atleast_2d(axes)
    for r, case in enumerate(cases):
        res, fr, arch = case_geometry(case)
        cs = crest_S(res, fr)
        st = args.station if args.station is not None else FIG_STATION[case]
        O = json.loads(Path(f"cases/{case}_outlines.json").read_text())
        rec = O["stations"].get(str(st))
        vols = {"pre": geom.load_volume(f"analysis/{case}/pre_cal.nii.gz"),
                "post": geom.load_volume(f"cbct_blinded/{case}/post_reg.nii.gz")}
        for c, which in enumerate(["pre", "post"]):
            img, ext, _u, _s = section(vols[which], arch, st, cs)
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=GREY[0], vmax=GREY[1], origin="lower",
                      extent=ext, aspect="equal", interpolation="nearest")
            if rec:
                V = np.array(rec[f"{which}_vertices_us_mm"], float)
                if len(V):
                    ax.plot(np.append(V[:, 0], V[0, 0]), np.append(V[:, 1], V[0, 1]),
                            color="#56B4E9" if which == "pre" else "#E69F00", lw=1.3)
            for yv, ls, nm in ((res["roi"]["Y_mm"][0], "--", "crest plane"),
                               (res["roi"]["Y_mm"][1], ":", "10 mm apical")):
                pl = plane_in_section(arch, st, fr, yv, (ext[0], ext[1]))
                if pl:
                    ax.plot(pl[0], pl[1], ls=ls, color="#e6e6e6", lw=1.1)
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_title(f"{case} station {st} — {'pre-operative' if which=='pre' else 'healed'}",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        if rec:
            P = us_to_local(rec["pre_vertices_us_mm"], arch, st, fr)
            bucc = P[P[:, 2] >= fr.offset[2]]
            report[case] = {
                "station": st,
                "traced_buccal_crest_Y_mm": round(float(bucc[:, 1].min()), 2) if len(bucc) else None,
                "window_crest_plane_Y_mm": res["roi"]["Y_mm"][0],
                "window_apical_plane_Y_mm": res["roi"]["Y_mm"][1],
                "traced_crest_coronal_to_window_mm":
                    round(float(res["roi"]["Y_mm"][0] - bucc[:, 1].min()), 2) if len(bucc) else None,
            }
        del vols
    fig.tight_layout()
    png = out / "QC_outline_overlay_2026-09-16.png"
    fig.savefig(png, dpi=200, facecolor="white")
    (out / "QC_outline_overlay_2026-09-16.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"wrote {png}")


if __name__ == "__main__":
    main()

