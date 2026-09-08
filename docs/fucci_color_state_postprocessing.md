# Fucci Color-State Postprocessing

This layer is additive to the daywise master pipeline. It reads one explicit
master-run directory and writes under
`<run_dir>/postprocess/fucci_color_state/`; it does not alter matching,
extraction tables, or the master runner.

## Measurement

Raw vertical fluorescence residuals become heteroscedastic as brightness
changes. Red is used as the predictor because it provides the stable reference
axis for the session-specific green relationship. For each session, the modal
green-vs-red backbone is fit from all ratio-valid native ROIs, not only complete
tracks:

```text
predicted_green = modal_intercept + modal_slope * red
log2_green_over_expected = log2(green / predicted_green)
eclipse_ratio = green / red
log2_eclipse_ratio = log2(green / red)
```

The fit uses an OLS initialization followed by a deterministic Gaussian-kernel
modal objective in standardized-red coordinates. The v1 bandwidth is
`0.6 * (1.4826 * MAD(OLS residuals))`, with ordinary residual SD used only when
the initial MAD is exactly zero and recorded in diagnostics. No epsilon is
added to the log residual.

## Dead reference and states

For each Dead-control session, the robust spread is
`1.4826 * median(abs(x - median(x)))`. Session spreads are reduced to a median
per mouse, then averaged with equal mouse weight. Fucci-Tri is divided by this
frozen Dead-derived scale; it is not re-centered or re-scaled using the target
mouse.

```text
Z_E = eclipse_deviation_log2 / dead_reference_robust_sd_log2
```

The public raw ratio fields are `eclipse_ratio` and `log2_eclipse_ratio`; the
public score is `eclipse_z` (also written as `Z_E`). The current construct
uses green as reporter and red as reference. Future constructs may map
different fluorescence channels to reporter and reference, while the ECLIPSE
names remain fluorophore-agnostic. The five bins are `strong_low` (`Z_E < -2`),
`low_transition` (`-2 <= Z_E < -1`), `middle` (`-1 <= Z_E <= 1`),
`high_transition` (`1 < Z_E <= 2`), and `strong_high` (`Z_E > 2`). Only strong low, middle, and strong high are core states. Transition
zones therefore provide hysteresis for entry events without being forced into a
core state.

## Workflow

Build one reference from at least two explicit Dead master runs:

```bash
python postprocessing/run_fucci_build_dead_reference.py \
  --reference-run-dir /path/to/Fucci-Dead_1/master_run \
  --reference-run-dir /path/to/Fucci-Dead_2/master_run \
  --output-dir /path/to/fucci_dead_color_reference_v1
```

Apply it to a target run:

```bash
python postprocessing/run_fucci_color_state.py \
  --run-dir /path/to/Fucci-Tri_1/master_run \
  --reference /path/to/fucci_dead_color_reference_v1/fucci_dead_color_reference.json
```

Then build color-Z trajectories, complete-case PCA, and entry-event summaries:

```bash
python postprocessing/run_fucci_state_analysis.py \
  --color-state-dir /path/to/Fucci-Tri_1/master_run/postprocess/fucci_color_state \
  --event-min-sessions 8 \
  --event-axis elapsed_days
```

For a non-canonical copied output, add `--run-dir /path/to/master_run`.

Events are detected from compressed core-state sequences, so
`middle -> low_transition -> low` is one middle-to-low entry. The full aligned
table retains all observations; default summaries and plots use the first event
of each direction per ROI. Both elapsed-day and session-index coordinates are
exported.

## Outputs and provenance

Normalization, trajectory, PCA, and event outputs are kept in separate
subdirectories. PCA centers complete-case columns only, does not variance-scale,
and uses NumPy SVD with a deterministic largest-loading-positive sign rule.
Every run records the source master/extraction hashes, input table hashes,
reference JSON hash, fit settings, thresholds, and row counts in
`run_manifest.json`. Input resolution is structural under the explicit run
directory, so copied runs do not depend on stale absolute manifest paths.
The trajectory/PCA filenames retain `color_z` during this transition, but their
feature column and public scored coordinate are `eclipse_z`.
