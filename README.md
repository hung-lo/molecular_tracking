# molecular_tracking

Longitudinal two-photon ROI identity tracking and molecular-state analysis.

`molecular_tracking` provides a reproducible project workflow for tracking the same segmented ROIs across longitudinal two-photon imaging sessions, extracting per-session fluorescence measurements, and carrying those identities into downstream Fucci/ECLIPSE analyses.

The **daywise workflow is the primary workflow**. The older weekly/explicit-path matcher is retained in the repository only for historical reproducibility and is no longer the recommended analysis path.

A central design rule is:

> Prefer a missing observation to an incorrect biological identity.

## Pipeline at a glance

```text
Raw ThorImage acquisitions + Experiment.xml
                │
                ▼
      Project catalog + pre-QC
      - acquisition discovery
      - PMT / laser-power checks
      - analysis_eligible gate
                │
                ▼
       Validated session manifest
       - 1050 nm primary
       - 920 nm optional
       - _vol10 excluded from quantitative analysis
                │
                ▼
       Daily registered images + masks
                │
                ▼
       Daywise matcher
       1. affine-overlap candidates
       2. high / balanced assignments
       3. spatial-graph refinement
       4. graph tracks used as final assignment
                │
                ▼
       Post-matching QC
       - registration / transform QC
       - cycle / gap checks
       - graph-vs-balanced agreement
       - segmentation / extraction eligibility
                │
                ▼
       Matched ROI extraction
       + quick plots / raw-space review
                │
                ├──────────────► Matcher evaluator
                │                - read-only endpoint diagnosis
                │                - pseudo-gap benchmark
                │
                └──────────────► Optional endpoint stitcher
                                 - conservative, state-blind
                                 - benchmark-gated
                                 - separate stitched identity view
```

## Quick start

### 1. Configure the project

Raw ThorImage data are treated as read-only inputs. Copy the example project config and edit the raw-data and derivatives roots:

```bash
cp config/project.example.toml config/project.local.toml
```

The raw root may use either the historical grouped layout or the flat acquisition-staging layout supported by the catalog builder.

### 2. Install

Python 3.11 or newer is required.

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
pytest tests/
```

Core Python dependencies are declared in `pyproject.toml`. Registration/segmentation environments used to create upstream daily images and masks may require additional image-analysis packages.

### 3. Build and validate the data catalog

Always inspect the catalog before launching a longitudinal analysis:

```bash
python tools/build_data_catalog.py \
  --project-config config/project.local.toml \
  --dry-run

python tools/build_data_catalog.py \
  --project-config config/project.local.toml \
  --strict
```

The catalog step discovers acquisitions, parses ThorImage metadata, runs acquisition-settings QC, and writes deterministic catalog/QC artifacts under the derivatives root.

### 4. Build the session manifest

Example for the primary 1050-nm Fucci analysis:

```bash
python tools/build_session_manifest.py \
  --project-config config/project.local.toml \
  --mouse-id Fucci-Tri_1 \
  --laser-nm 1050
```

If the required registered images and masks are not available yet, the builder writes a planning manifest rather than pretending the dataset is analysis-ready.

### 5. Run the primary daywise pipeline

```bash
.venv/bin/python core/run_daywise_master_pipeline.py \
  --project-config config/project.local.toml \
  --mouse-id Fucci-Tri_1 \
  --laser-nm 1050
```

For a small contiguous test run:

```bash
# earliest five sessions
.venv/bin/python core/run_daywise_master_pipeline.py \
  --project-config config/project.local.toml \
  --mouse-id Fucci-Tri_1 \
  --laser-nm 1050 \
  --sessions first:5

# most recent five sessions
.venv/bin/python core/run_daywise_master_pipeline.py \
  --project-config config/project.local.toml \
  --mouse-id Fucci-Tri_1 \
  --laser-nm 1050 \
  --sessions last:5
```

`--sessions first:N` and `--sessions last:N` create a run-local subset and never modify the validated project manifest.

---

# Current tracking architecture

## 1. Pre-analysis data QC

Acquisition QC happens before sessions are allowed into the quantitative longitudinal manifest.

`tools/build_data_catalog.py` parses each acquisition's ThorImage `Experiment.xml` and records:

- PMT A and PMT B gain;
- 920-nm and 1050-nm Pockels start/stop values;
- acquisition geometry and z-stack metadata;
- averaging and software metadata;
- per-acquisition `settings_qc_pass`;
- a human-readable `settings_qc_reason`;
- a machine-readable `settings_qc_status` (`pass`, `fail`, `not_configured`, or `not_applicable`);
- final `analysis_eligible` status.

### Fucci acquisition settings

The currently configured Fucci acquisition expectations are:

| Mouse | Pipeline | PMT A | PMT B | 920-nm power | 1050-nm power |
|---|---|---:|---:|---:|---:|
| `Fucci-Tri_1` | enabled | 10 | 10 | 50 | 50 |
| `Fucci-Tri_2` | excluded — poor FoV quality | n/a | n/a | n/a | n/a |
| `Fucci-Tri_3` | enabled | 10 | 10 | 60 | 60 |
| `Fucci-Tri_4` | enabled | 10 | 10 | 70 | 70 |
| `Fucci-Dead_1` | enabled | 10 | 10 | 70 | 70 |
| `Fucci-Dead_2` | enabled | 10 | 10 | 70 | 70 |

For an active laser, both Pockels start and stop values must match the configured expected power. The selected laser is always treated as active, so an accidentally zero or incorrect selected-laser power fails QC.

A Fucci acquisition that fails this check is marked `analysis_eligible = False` and is excluded from manifest generation.

`Fucci-Tri_2` is explicitly disabled from the longitudinal pipeline because of poor FoV quality. Its raw acquisitions remain in the catalog for provenance, but it is omitted from analysis manifests and the primary QC plot; it is not reported as an acquisition-settings failure.

The Fucci workflow also **fails closed on old catalogs** that do not contain the acquisition-QC fields. Rebuild them with `tools/build_data_catalog.py` rather than silently analyzing data with unknown acquisition settings.

The catalog writes QC artifacts including:

```text
_catalog/
├── acquisitions.generated.csv
├── sessions.generated.csv
├── acquisition_settings_qc.csv
├── acquisition_settings_qc.png
├── acquisition_settings_qc.json
├── mice.validated.csv
└── validation_report.json
```

The JSON summary records pass/fail counts, excluded `_vol10` controls, unconfigured mice, and failed-session reasons for automated review.

The master run also checks selected-session consistency. PMT gains, selected-laser Pockels values, and averaging are compared across the selected sessions; setting changes can be promoted from warnings to a hard stop with the acquisition-consistency requirement.

### Project-level acquisition rules

For the current Fucci project:

- **1050 nm is the primary longitudinal analysis channel.**
- **920 nm is optional** and is used as a validation/cross-laser layer rather than as a requirement for the main 1050 trajectory analysis.
- Acquisitions containing `_vol10` are alignment-only and are excluded from quantitative manifests.
- `Fucci-Tri_2` is pipeline-excluded for poor FoV quality; all other listed Fucci mice are enabled.
- Raw acquisition folders are never modified by the pipeline.
- Generated products belong under the configured derivatives root.

---

## 2. Daywise matcher

The current matcher operates directly on daily segmented masks.

The master pipeline runs two linked stages:

1. **Affine-overlap matching**
2. **Spatial-graph refinement**

### Affine-overlap stage

The affine-overlap matcher:

- estimates coarse session-to-session displacement from binary occupancy;
- fits a restricted transform between sessions;
- transforms ROI geometry into a common physical space;
- constructs a shared candidate table;
- solves separate one-to-one `high` and `balanced` assignments.

`high` is the strict branch. `balanced` provides the broader baseline candidate assignment used by the graph refinement and for agreement auditing.

### Spatial-graph refinement

The spatial-graph matcher is a conservative refinement of the affine result.

It:

- reuses the baseline pairwise candidates;
- uses strict `high` matches as anchors;
- limits refinement to candidate relationships admitted by the baseline balanced branch;
- evaluates local geometric consistency after transforming centroids into common physical space;
- produces a deterministic graph assignment.

The graph score is geometric; it is not a learned classifier.

### Final assignment used by the master pipeline

In the current master workflow, the **graph result is the final longitudinal assignment**.

The affine `balanced` result remains valuable as an independent provenance/comparison branch. Final graph tracks are annotated as:

- `consensus`: every accepted graph edge is also present in the balanced assignment;
- `graph_only`: at least one accepted graph edge differs from balanced;
- `unmatched`: singleton ROI with no accepted longitudinal edge.

This agreement information is propagated into compatible extraction tables so downstream analyses can distinguish conservative graph/affine consensus tracks from identities that depend on graph-only decisions.

Important matching outputs include:

```text
session_manifest_resolved.csv
roi_features.csv

pairwise_transforms.csv
pairwise_summary.csv

pairwise_matches_high.csv
pairwise_matches_balanced.csv
pairwise_matches_graph.csv

tracks_high.csv
tracks_balanced.csv
tracks_graph.csv

track_edges_*.csv
cycle_consistency_*.csv
cycle_edge_checks_*.csv

graph_match_changes.csv
pairwise_matches_graph_agreement.csv
accepted_graph_edges_with_agreement.csv
graph_affine_agreement_track_summary.csv
graph_affine_agreement_summary.json
```

See:

- `docs/daywise_roi_matching_algorithm.md`
- `docs/daywise_spatial_graph_matching.md`
- `docs/output_schema.md`

---

## 3. Post-matching QC

Matching QC is generated automatically after matching unless explicitly skipped.

It is a **review and failure-detection layer**, not a second source of biological identity.

The QC bundle is designed to surface:

- pairwise examples with poor geometric alignment;
- weak or fallback transforms;
- cycle inconsistencies;
- one-gap bridges and other unusual track connections;
- graph decisions that differ from the balanced affine branch;
- suspicious segmentation geometry;
- extraction/trajectory eligibility issues.

By default, QC findings can be inspected without deleting the matching output. Runs that must fail when the QC stage fails can enable the corresponding strict QC requirement.

The master pipeline also exposes segmentation/trajectory gates, including configurable volume limits, z-depth limits, edge exclusion, track-relative volume checks, minimum usable-session requirements, and internal-gap/coverage rules.

For interpretation of the matching QC bundle, see:

- `docs/qc_interpretation.md`
- `docs/daywise_pipeline.md`

---

## 4. Matched ROI extraction

After graph-track construction and QC, the master pipeline performs matched ROI intensity extraction.

The extraction layer:

- keeps structural missingness distinct from failed intensity extraction;
- writes all eligible ROI-session observations as well as complete-track subsets;
- applies session-level normalization using signal-valid native ROIs from that session;
- keeps trajectory eligibility separate from the normalization fit;
- propagates graph-vs-affine agreement provenance;
- generates quick plots and native raw-space review panels.

The primary identity source remains `tracks_graph.csv` from the canonical matching directory.

Downstream Fucci/ECLIPSE state calculations are postprocessing steps and do not feed back into identity matching.

---

# Matcher evaluator

`matching/evaluate_daywise_tracking.py` is a **read-only diagnostic layer** for an existing matching run.

Its purpose is to understand why canonical tracks end and whether a plausible later ROI exists.

Typical run:

```bash
python matching/evaluate_daywise_tracking.py \
  --match-dir /path/to/matching \
  --output-dir /path/to/matching/evaluation \
  --policy graph \
  --lookahead 3 \
  --search-radius-um 15 \
  --crop-radius-um 45 \
  --z-radius 1 \
  --max-review-panels 100
```

Optional extraction and state tables can be supplied for review/enrichment, but they do not alter canonical matching.

The evaluator asks questions such as:

- Did the same canonical track already recover after a one-session gap?
- Is there a nearby later track start?
- Is the nearby object only a singleton?
- Is there no segmented ROI near the predicted position?
- Is the endpoint close to an unreliable image/FOV boundary?
- Is registration/transform support unreliable?
- Are multiple later candidates geometrically plausible?

It also generates raw-space review panels and a synthetic pseudo-gap benchmark using trusted continuous tracks.

The evaluator **does not**:

- rerun the matcher;
- change affine, graph, or gap thresholds;
- edit canonical track IDs;
- create canonical `t -> t+3` links;
- interpolate missing masks;
- use molecular state to decide identity.

Automatic triage is intentionally conservative. Biological interpretations such as `segmentation_dropout`, `matcher_fragment`, or `true_disappearance` remain review labels rather than automatic declarations.

See `docs/daywise_matcher_evaluator.md`.

---

# Conservative endpoint stitcher

`matching/run_endpoint_stitching.py` is an **optional post-hoc identity-recovery stage** built on top of the canonical matcher and evaluator.

It is designed for the specific case where a canonical track ends and a plausible continuation begins later as a different canonical track.

Typical proposal/benchmark run:

```bash
python matching/run_endpoint_stitching.py \
  --match-dir /path/to/matching \
  --evaluation-dir /path/to/matching/evaluation \
  --output-dir /path/to/endpoint_stitching \
  --policy graph \
  --max-gap-sessions 3 \
  --search-radius-um 15 \
  --profile conservative_v1 \
  --benchmark-replicates 20 \
  --benchmark-cases-per-gap 50 \
  --random-seed 0 \
  --max-review-panels 100 \
  --overwrite
```

The stitcher is deliberately separate from canonical matching.

### Safety properties

The stitcher:

- only considers true canonical endpoints and later starts of different tracks;
- supports gaps of 1–3 sessions;
- uses reciprocal spatial ranking, transform reliability, local anchor residuals, source-history consistency, future-target consistency, and canonical track QC;
- uses a global one-to-one assignment so a source or target cannot be reused;
- does not interpolate masks or observations;
- preserves the original canonical track UIDs in provenance tables;
- refuses same-session collisions and checks observation conservation;
- never writes back into the canonical matching directory.

### State-blind identity assignment

Molecular state is intentionally excluded from identity decisions.

ECLIPSE values, red/green intensity, and `color_z` may be carried as review provenance, but they are not used for candidate tiering or assignment cost. This prevents the identity algorithm from preferentially linking cells because their molecular trajectories happen to look similar.

### Benchmark gate

Before derived stitched tracks are written, the stitcher runs a pseudo-fragment benchmark that splits trusted continuous canonical tracks and asks whether the same algorithm can recover their known identities while rejecting negative/collision controls.

The benchmark has strict precision and false-positive guardrails. If the guardrail is not met, stitched track files are blocked unless the user explicitly overrides that write guard.

To request the derived stitched identity view after the benchmark:

```bash
python matching/run_endpoint_stitching.py \
  ... \
  --write-stitched-tracks
```

This may produce:

```text
tracks_graph_stitched.csv
track_uid_stitch_map.csv
```

alongside proposals, assignments, benchmark tables, review manifests, PNG panels, summaries, and provenance logs.

**Canonical graph tracks remain unchanged.** Stitched tracks are a separate derived view and should not become the primary scientific identity source until the frozen stitch profile has been validated on an independent/hold-out dataset.

See `docs/daywise_endpoint_stitching.md`.

---

# Identity layers and what is canonical

The repository intentionally keeps identity layers separate:

| Layer | Purpose | Changes canonical tracks? |
|---|---|---|
| Affine `high` / `balanced` | baseline pairwise matching and comparison | No |
| Graph matcher | primary final assignment in the master workflow | Produces canonical graph tracks |
| Matching QC | detect/review suspicious geometry and track structure | No |
| Evaluator | diagnose track endings and benchmark recoverability | No |
| Endpoint stitcher | optional recovery of fragmented identities | No; writes a separate derived identity view |
| Fucci/ECLIPSE postprocessing | molecular-state analysis | No |

This separation makes it possible to audit exactly which conclusions depend on canonical graph matching, graph-only decisions, or optional post-hoc stitching.

---

# Repository layout

```text
molecular_tracking/
├── config/          project configuration templates
├── core/            catalog, metadata/QC, extraction, and master pipeline
├── matching/        affine/graph matching, matching QC, evaluator, stitcher
├── plotting/        quick plots and raw-space validation figures
├── postprocessing/  Fucci/ECLIPSE and downstream trajectory analyses
├── metadata/        project metadata
├── validation/      validation utilities/workflows
├── tools/           project/catalog/manifest helper CLIs
├── tests/           regression and behavior tests
├── docs/            algorithm and output documentation
├── examples/        example manifests/configuration inputs
└── notebooks/       intentionally retained demos/reference notebooks
```

## Main entry points

| Task | Entry point |
|---|---|
| Build acquisition catalog + pre-QC | `tools/build_data_catalog.py` |
| Build project session manifest | `tools/build_session_manifest.py` |
| Run primary daywise workflow | `core/run_daywise_master_pipeline.py` |
| Run graph matching directly | `matching/run_daywise_graph_matching.py` |
| Evaluate canonical endpoints | `matching/evaluate_daywise_tracking.py` |
| Run conservative endpoint stitching | `matching/run_endpoint_stitching.py` |
| Fucci/ECLIPSE postprocessing | `postprocessing/` |

---

# Documentation

Start with:

- `docs/multi_mouse_workflow.md` — project/catalog/manifest workflow
- `docs/daywise_roi_matching_algorithm.md` — affine-overlap baseline
- `docs/daywise_spatial_graph_matching.md` — spatial-graph refinement
- `docs/qc_interpretation.md` — matching QC interpretation
- `docs/daywise_pipeline.md` — extraction, completeness, and trajectory rules
- `docs/daywise_matcher_evaluator.md` — endpoint evaluator
- `docs/daywise_endpoint_stitching.md` — conservative stitcher and benchmark guardrail
- `docs/output_schema.md` — CSV/output schemas
- `docs/fucci_color_state_postprocessing.md` — Fucci/ECLIPSE postprocessing
- `docs/troubleshooting.md` — common failure modes

---

# Data and reproducibility

This repository is intended to remain code-first.

- Keep raw imaging data outside the repository.
- Treat raw ThorImage acquisitions as read-only.
- Keep generated derivatives, figures, CSV exports, caches, and checkpoints out of source control unless they are intentionally curated examples.
- Record run configuration and provenance with each analysis.
- Prefer deterministic, auditable outputs over hidden notebook state.
- Add or update tests whenever reusable matching/QC behavior changes.

---

# Legacy workflow

The historical weekly/explicit-path matcher and associated scripts remain in the repository for reproducibility of older analyses.

They are **not the recommended workflow for new longitudinal datasets** and are intentionally not documented as a parallel first-class pipeline here.

For current work, use:

```text
project catalog / pre-QC
        →
daywise graph-aware master pipeline
        →
matching QC
        →
read-only evaluator
        →
optional benchmark-gated endpoint stitching
        →
downstream molecular-state analysis
```

If an older dataset must be reproduced exactly, use the historical scripts and the documentation or commit corresponding to that analysis rather than mixing weekly and current daywise identity products.
