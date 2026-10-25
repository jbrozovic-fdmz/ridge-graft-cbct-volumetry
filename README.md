# ridge-graft-cbct-volumetry

Analysis code for the article "Non-Contained Maxillary Ridge Defects Augmented with 50:50 Synthetic–Autogenous and Xenogeneic–Autogenous Composite Grafts: Four Cases with CBCT Volumetry".

**Requirements:** Python 3.12.3 with NumPy 2.5.1, SciPy 1.18.0, scikit-image 0.26.0, matplotlib 3.11.0 and SimpleITK 2.5.5; CloudCompare 2.13.2 on the PATH as `CloudCompare` (cross-check in step 2 only).

**Case codes:** article S1, X1, S2, X2 = files S1, S3, S2, S4.

**Inputs:** `cbct_blinded/<C>/pre.nii.gz`, `post.nii.gz`; `cases/<C>.json`, `<C>_readerB.json`, `<C>_arch.json`, `<C>_outlines.json`; and the records `analysis/<C>/fixation_pins_2026-09-13.json`, `FINAL_ARCH_20260907/fixation_pins_2026-09-13.json`, `FINAL_ARCH_20260907/FROZEN_RESULTS.json`, `FINAL_ARCH_20260907/containment_NULLCORRECTED.json`, `FINAL_ARCH_20260907/registration_sitebox_rule_2026-09-14.json`, `analysis/<C>/sitebox_sensitivity_2026-09-14/landmark_box/transform_resample2_A_leaf0.tfm` and `analysis/S4/audit_20260906/S4_good_full.tfm` (`<C>` = S1, S2, S3, S4).

**Run order,** every command from the repository root (`S` = `analysis/scalefix_2026-10-06`, `V` = `analysis/revision_2026-10`):

1. `python3 analysis/selftest.py`
2. `python3 analysis/run_case.py <C>`, then `python3 analysis/run_case.py <C> --reader B --skip-registration`; `python3 analysis/specification_curve.py <C>`; `python3 analysis/outline_compare.py <C>`
3. `python3 analysis/pin_artefact_rule_2026-09-14.py --declare`, `--profile <C>`, `--collect`
4. `mkdir -p S/<C>`; `python3 S/sfx_frozen_transform.py <C>`; `python3 S/sfx_register.py <C> mask`; `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1 python3 S/sfx_register.py <C> primary`; `python3 S/sfx_register.py <C> rigid`, `uncert`, `affine`, `uncert_affine`, `bounds`; S2 also `bounds_affine` and `materialize affine`
5. `python3 S/sfx_metalctl.py <C> <reg> height`, with `--thr` for the primary `<reg>` = similarity (S2: affine), then scalelo and scalehi (S2: affinelo, affinehi), and rigid for S1; `python3 S/sfx_measure.py <C> <reg> height --thr` for the primary `<reg>`; `python3 S/sfx_members.py <C> <reg> height_metalctl` (S3: `height`). The original registration with the pin correction: `python3 S/sfx_measure.py <C> frozen old --thr --tag _final2`
6. `python3 S/sfx_mirror.py build_selection primary`, then in `S/mirror_primary`: `python3 analysis/null_window_check.py <C>`, `python3 analysis/outline_compare.py <C>`, `python3 analysis/linear_vertical_gain.py`
7. `python3 S/sfx_assemble.py --variant metalctl`
8. `python3 V/original_envelopes_2026-10-24.py`; `python3 V/threshold_sweep_net_2026-10-24.py --case <C>`, `--collect`; `python3 V/threshold_sweep_frozen_background_2026-10-24.py`; `python3 V/calibration_validation_2026-10-24.py --case <C> --stage grey`, then `--stage volume`, then `--collect`; `python3 V/background_offset_effect_2026-10-24.py --volumes`, `--field <C> <reg>`, `--collect`; `python3 V/defect_dimensions_preop_2026-10-24.py --case <C>`, `--collect`; then the 0.75 mm control limit in a copy of the project root (`<copy>`) that holds only the code, `cases/`, `FINAL_ARCH_20260907/fixation_pins_2026-09-13.json`, `containment_NULLCORRECTED.json` and `FROZEN_RESULTS.json`, `analysis/S4/audit_20260906/S4_good_full.tfm`, and per case `analysis/<C>/results.json`, `readerB/results.json`, `pre_cal.nii.gz`, `post_cal.nii.gz`, `registration.json` and `fixation_pins_2026-09-13.json`, `cbct_blinded/<C>/post_reg.nii.gz`, the step-4 files `S/<C>/post_reg_*`, `reg_*`, `uncert_*`, `materialize_*` and `crowns_*_v3.json`, and the step-6 files `S/mirror_primary/FINAL_ARCH_20260907/linear_vertical_gain.json` and `S/mirror_primary/analysis/<C>/outline_compare.json` and `null_window_check.json`; in `<copy>`, step 3, then `mkdir -p S/<C>` and `python3 S/sfx_frozen_transform.py <C>`, then `python3 V/control_limit_075_2026-10-24.py --engine <C> <reg> height --metalctl` for the step-5 runs (S3 similarity without `--metalctl`) and `--engine <C> frozen old`, then `python3 S/sfx_members.py` as in step 5, `python3 S/sfx_assemble.py --variant metalctl` and `python3 V/control_limit_075_2026-10-24.py --speccurve <C>`; then, from the project root, `--collect <copy>`
9. Traced areas: `python3 S/traced_above_plane_2026-10-08_inline.py`, run in the figure tree that holds the Figure 4 section images

The other `pin_artefact_*` and `pin_prominence_*` scripts are the earlier steps of the pin correction; steps 3 and 5 load their functions.
