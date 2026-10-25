from __future__ import annotations

import sys
import traceback

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

sys.path.insert(0, "analysis")
from pipeline import calibrate, geom, volume

VX = 0.24
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("  PASS  " if ok else "  FAIL  ") + name + ("   " + detail if detail else ""))
    return ok


def test_marching_cubes_axis_order():
    v = np.zeros((40, 50, 60), np.float32)
    v[5:10, 20:25, 45:50] = 900.0
    V, _, _, _ = measure.marching_cubes(v, level=450.0, spacing=(VX, VX, VX))
    c = V.mean(axis=0) / VX
    want = np.array([7.0, 22.0, 47.0])
    check("marching_cubes returns columns in input-array order (a0, a1, a2)",
          np.allclose(c, want, atol=0.6), f"centroid={np.round(c, 2)} expected~{want}")


def test_outward_normals():
    n = 48
    g = np.mgrid[0:n, 0:n, 0:n].astype(float)
    r = np.sqrt(((g - (n - 1) / 2.0) ** 2).sum(axis=0))
    field = ndi.gaussian_filter(np.where(r < 15, 900.0, 0.0).astype(np.float32), 0.6)
    V, F = geom.iso_surface(field, VX)
    cent, _, nout = geom.face_geometry(V, F)
    radial = cent - cent.mean(axis=0)
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    frac = float(((nout * radial).sum(axis=1) > 0).mean())
    check("geom.iso_surface returns OUTWARD-pointing face normals", frac > 0.99,
          f"outward fraction={frac:.3f}")


def test_frame_is_a_rotation_and_round_trips():
    f = geom.build_frame(platform_SAL=[10, 20, 30], apex_SAL=[10, 31, 30],
                         buccal_hint_SAL=[0, 0, 1])
    det = float(np.linalg.det(f.R))
    P = np.array([[10.0, 20.0, 30.0], [12.5, 24.0, 33.0], [-3.0, 7.0, 1.5]])
    back = f.to_anatomical(f.to_local(P))
    check("LocalFrame is a proper rotation (det = +1)", abs(det - 1) < 1e-9, f"det={det:.9f}")
    check("LocalFrame round-trips anatomical -> local -> anatomical",
          np.allclose(back, P, atol=1e-9),
          f"max err={np.abs(back - P).max():.2e} mm")
    cross = np.cross(f.R[0], f.R[1])
    check("local frame satisfies X x Y = Z (right-handed)", np.allclose(cross, f.R[2], atol=1e-9))
    q = f.to_local(np.array([[10.0, 20.0, 30.0]]))[0]
    check("implant platform maps to the ROI's coronal face", abs(q[1] - 15.0) < 1e-9,
          f"Y={q[1]:.6f}")
    qa = f.to_local(np.array([[10.0, 31.0, 30.0]]))[0]
    check("implant apex maps to positive (apical) Y", qa[1] > q[1], f"Y_apex={qa[1]:.3f}")


def build_phantom(H=3.0, sigma=2.5, n=180, z0=70, implant=False):
    a0, a1, a2 = np.mgrid[0:n, 0:n, 0:n].astype(np.float32)
    x = a0 * VX
    y = a1 * VX
    z = a2 * VX
    cx = cy = n * VX / 2.0
    h = H * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))
    zsurf = z0 * VX
    w = 0.35 * VX
    pre = (450.0 * (1.0 + np.tanh((zsurf - z) / w))).astype(np.float32)
    post = (450.0 * (1.0 + np.tanh((zsurf + h - z) / w))).astype(np.float32)
    imp_vox = None
    if implant:
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        cyl = (r <= 1.5) & (z > zsurf - 4.0) & (z <= zsurf + 1.0)
        post = np.where(cyl, 3200.0, post).astype(np.float32)
        imp_vox = np.argwhere(cyl)
    return pre, post, h, imp_vox


def test_phantom_volume_recovery():
    H, sigma, n, z0 = 3.0, 2.5, 180, 70
    pre, post, h, _ = build_phantom(H, sigma, n, z0)
    truth_infinite = H * 2 * np.pi * sigma ** 2

    pre_f = geom.bone_field(pre)
    post_f = geom.bone_field(post)
    Vp, Fp = geom.iso_surface(pre_f, VX)
    Vq, Fq = geom.iso_surface(post_f, VX)
    pre_pts = geom.densify(Vp[Fp])
    post_pts = geom.densify(Vq[Fq])

    f = geom.build_frame(platform_SAL=[n * VX / 2, n * VX / 2, z0 * VX],
                         apex_SAL=[n * VX / 2, n * VX / 2 + 10, z0 * VX],
                         buccal_hint_SAL=[0, 0, 1])
    pl = f.to_local(pre_pts)
    ql = f.to_local(post_pts)

    half = 10.0
    xr = (15.0 - half, 15.0 + half)
    yr = (15.0 - half, 15.0 + half)
    truth = float(h.max(axis=2).sum() * VX * VX) if False else truth_infinite

    for step in (0.1, 0.2, 0.3):
        m = volume.measure_window(pl, ql, xr, yr, step=step)
        err = 100 * (m["gain_mm3"] - truth) / truth
        check(f"phantom volume recovered at {step} mm grid (truth {truth:.1f} mm3)",
              abs(err) < 3.0, f"measured {m['gain_mm3']:.1f} mm3, error {err:+.2f} %")
        if step == 0.2:
            check("phantom height field has full coverage", m["coverage"] > 0.999,
                  f"coverage={m['coverage']}")
            check("phantom peak height recovered",
                  abs(m["max_thickness_mm"] - H) < 0.1,
                  f"max {m['max_thickness_mm']:.3f} mm vs {H} mm")


def test_phantom_implant_subtraction():
    H, sigma, n, z0 = 3.0, 2.5, 180, 70
    pre, post, h, imp_vox = build_phantom(H, sigma, n, z0, implant=True)
    pre_f = geom.bone_field(pre)
    post_f = geom.bone_field(post)
    Vp, Fp = geom.iso_surface(pre_f, VX)
    Vq, Fq = geom.iso_surface(post_f, VX)
    f = geom.build_frame(platform_SAL=[n * VX / 2, n * VX / 2, z0 * VX],
                         apex_SAL=[n * VX / 2, n * VX / 2 + 10, z0 * VX],
                         buccal_hint_SAL=[0, 0, 1])
    pl = f.to_local(geom.densify(Vp[Fp]))
    ql = f.to_local(geom.densify(Vq[Fq]))
    m = volume.measure_window(pl, ql, (5.0, 25.0), (5.0, 25.0), step=0.2)
    sub = volume.implant_volume_in_envelope(imp_vox, f, m["_Hpre"], m["_Hpost"],
                                            m["_x_edges"], m["_y_edges"], VX)

    rr = np.linspace(0, 1.5, 2000)
    hh = H * np.exp(-(rr ** 2) / (2 * sigma ** 2))
    truth = float(np.trapezoid(np.minimum(hh, 1.0) * 2 * np.pi * rr, rr))
    got = sub["implant_in_envelope_mm3"]
    err = 100 * (got - truth) / truth
    check("implant volume inside the gain envelope recovered",
          abs(err) < 10.0, f"measured {got:.2f} mm3 vs analytic {truth:.2f} mm3 ({err:+.1f} %)")


def test_height_field_selects_the_buccal_branch():
    xe = np.array([14.9, 15.1])
    ye = np.array([14.9, 15.1])
    pts = np.array([[15.0, 15.0, 3.0], [15.0, 15.0, 9.0],
                    [15.0, 15.0, 16.9], [15.0, 15.0, 17.1]])
    H = geom.height_field(pts, xe, ye)
    check("height_field ignores deeper surfaces and averages the buccal branch",
          abs(H[0, 0] - 17.0) < 1e-9, f"got {H[0, 0]}")
    Hm = geom.height_field(pts, xe, ye, mode="max")
    check("height_field mode='max' reproduces CloudCompare -PROJ MAX",
          abs(Hm[0, 0] - 17.1) < 1e-9, f"got {Hm[0, 0]}")


def test_despike_removes_dropouts_not_anatomy():
    H = np.full((11, 11), 15.0)
    H[5, 5] = 9.0
    Ha, na = geom.despike(H.copy())
    check("despike removes an isolated dropout", bool(np.isnan(Ha[5, 5])) and na == 1,
          f"{na} cell(s) rejected")
    Hb = np.full((11, 11), 15.0)
    Hb[:, 6:] = 10.0
    Hc, nb = geom.despike(Hb.copy())
    check("despike preserves a coherent 5 mm step", bool(np.isfinite(Hc).all()) and nb == 0,
          f"{nb} cell(s) rejected")


def test_cylinder_radius():
    from pipeline.landmarks import _cylinder_radius
    rng = np.random.default_rng(3)
    R, Hh = 1.7, 11.0
    m = rng.random(60000)
    r = R * np.sqrt(m)
    th = rng.random(60000) * 2 * np.pi
    z = rng.random(60000) * Hh
    pts = np.column_stack([r * np.cos(th), r * np.sin(th), z])
    centroid = pts.mean(axis=0)
    axis = np.array([0.0, 0.0, 1.0])
    proj = (pts - centroid) @ axis
    got = _cylinder_radius(pts, centroid, axis, proj)
    check("implant radius recovered from a synthetic cylinder", abs(got - R) < 0.05,
          f"got {got:.3f} mm, true {R} mm")


def test_boolean_band_is_configurable():
    import inspect
    from pipeline import volume as _v
    sig = inspect.signature(_v.boolean_volume).parameters
    check("boolean_volume takes z_min/z_max rather than a fixed span",
          "z_min" in sig and "z_max" in sig and "z_span_mm" not in sig,
          f"parameters: {list(sig)}")


def _synthetic_scan(air, soft, bone, span_noise=40.0, seed=3, padding=None):
    rng = np.random.default_rng(seed)
    a = np.empty((60, 60, 60), np.float32)
    a[:] = rng.normal(soft, span_noise, a.shape)
    a[10:50, 10:50, 10:22] = rng.normal(air, span_noise, (40, 40, 12))
    a[24:36, 24:36, 30:42] = rng.normal(bone, span_noise, (12, 12, 12))
    if padding is not None:
        a[:6] = padding
        a[:8, :4] = padding * 0.8
    return a


def test_calibration_recovers_a_known_linear_map():
    for air, soft, tag in [(-1000., 60., "already calibrated"),
                           (-800., 10., "compressed (S3/S4 pre)"),
                           (-985., 177., "stretched (S3 post)"),
                           (0., 900., "no negative values at all")]:
        a = _synthetic_scan(air, soft, soft + 0.4 * (soft - air) + 900)
        ref = calibrate.grey_references(a)
        b = calibrate.calibrate_array(a, ref)
        got = calibrate.grey_references(b)
        check(f"calibration maps air->-1000, soft->+60 [{tag}]",
              abs(got["air"] - calibrate.AIR_HU) < 25 and abs(got["soft"] - calibrate.SOFT_HU) < 25,
              f"recovered air={got['air']:.0f}, soft={got['soft']:.0f}")


def test_calibration_is_a_pure_linear_map():
    a = _synthetic_scan(-800., 10., 1200.)
    ref = calibrate.grey_references(a)
    b = calibrate.calibrate_array(a, ref)
    v = a.ravel()[::37]
    w = b.ravel()[::37]
    A = np.vstack([v, np.ones_like(v)]).T
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    resid = float(np.abs(A @ coef - w).max())
    check("calibration is exactly linear (no clipping of tissue values)", resid < 1e-2,
          f"max residual {resid:.2e} grey levels")


def test_calibration_removes_the_padding_sentinel():
    a = _synthetic_scan(-800., 10., 1200., padding=-11486.)
    ref = calibrate.grey_references(a)
    check("padding sentinel is detected", ref["padding_value"] is not None,
          f"found {ref['padding_value']}")
    b = calibrate.calibrate_array(a, ref)
    check("padding and its halo are removed from the calibrated volume",
          b.min() >= calibrate.AIR_HU - 500.0 - 1e-3,
          f"calibrated min {b.min():.0f} (was {a.min():.0f})")


def test_calibration_detects_a_clipped_air_peak():
    a = _synthetic_scan(-1000., 60., 1300.)
    a = np.maximum(a, -1000.)
    ref = calibrate.grey_references(a)
    check("air peak found when the machine clips air to the lowest stored value",
          abs(ref["air"] - (-1000.)) < 30, f"air={ref['air']:.0f}")


def test_calibration_refuses_a_scan_it_cannot_calibrate():
    a = np.random.default_rng(0).normal(0., 50., (40, 40, 40)).astype(np.float32)
    try:
        calibrate.grey_references(a)
        check("calibration aborts on a scan with no air peak", False, "it returned a value")
    except SystemExit:
        check("calibration aborts on a scan with no air peak", True)


def main():
    print("PIPELINE SELF-TEST\n" + "=" * 78)
    print("\n[conventions]")
    for t in (test_marching_cubes_axis_order, test_outward_normals,
              test_frame_is_a_rotation_and_round_trips, test_height_field_selects_the_buccal_branch,
              test_cylinder_radius, test_boolean_band_is_configurable,
              test_despike_removes_dropouts_not_anatomy):
        try:
            t()
        except Exception:
            check(t.__name__, False, "EXCEPTION")
            traceback.print_exc()
    print("\n[grey-value calibration]")
    for t in (test_calibration_recovers_a_known_linear_map, test_calibration_is_a_pure_linear_map,
              test_calibration_removes_the_padding_sentinel,
              test_calibration_detects_a_clipped_air_peak,
              test_calibration_refuses_a_scan_it_cannot_calibrate):
        try:
            t()
        except Exception:
            check(t.__name__, False, "EXCEPTION")
            traceback.print_exc()
    print("\n[phantom with analytically known volume]")
    for t in (test_phantom_volume_recovery, test_phantom_implant_subtraction):
        try:
            t()
        except Exception:
            check(t.__name__, False, "EXCEPTION")
            traceback.print_exc()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

