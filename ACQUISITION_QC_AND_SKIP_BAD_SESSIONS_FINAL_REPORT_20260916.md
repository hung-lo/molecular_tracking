# Acquisition QC and skip-bad-sessions final report (2026-09-16)

baseline: `042f643`
ending commit: `bfdee4c`
focused tests: `32 passed` (acquisition QC/catalog)
full suite: `340 passed, 2 skipped`

Fresh read-only-derived catalog, QC CSV/JSON, and PNG:
`D:/_data/_newAAV_2026/molecular_tracking_derivatives/_qc_final_20260916/_catalog`

## VOL10

- `_vol10` rows detected: `170`
- `_vol10` rows excluded from QC: `170`
- `_vol10` rows plotted: `0`
- `_vol10` rows in manifests: `0`
- `vol100`/`vol105` false positives: `0`

## DISCOVERY

- Valid XML cannot be lost by folder-name heuristic: **YES**
- Missing/malformed analysis acquisitions remain auditable as row-ineligible records.
- Row-level failures are separate from fatal catalog errors; good sessions remain manifestable.

## ROW-LEVEL QC

- PMT A/B checked against 10: **YES**
- Active-laser start/stop checked: **YES**
- Inactive laser ignored: **YES**
- `analysis_included` and `analysis_eligible` remain separate.

## SKIPPING

- Bad analysis session remains auditable: **YES**
- Bad analysis session excluded from manifest: **YES**
- Unrelated good sessions continue: **YES**
- Fresh catalog fatal errors: `0`; row-ineligible records: `1`.

## UNCONFIGURED FUCCI

- Policy: fail fast at manifest generation with an explicit missing-configuration error; catalog rows are `not_configured`, `settings_qc_pass=NA`, and ineligible.
- Silent per-session exclusion: **NO**
- Real-data unconfigured rows: Fucci-Tri_2=`6`, Fucci-Tri_4=`2`.

## REAL DATA

| mouse | total rows | `_vol10` | analysis-included | QC pass/fail | eligible/ineligible | missing XML | malformed XML | eligible 1050 sessions | eligible 920 sessions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fucci-Tri_1 | 126 | 51 | 67 | 67/0 | 67/0 | 0 | 0 | 38 | 29 |
| Fucci-Tri_3 | 21 | 7 | 14 | 13/1 | 13/1 | 0 | 0 | 6 | 7 |
| Fucci-Dead_1 | 80 | 46 | 34 | 34/0 | 34/0 | 0 | 0 | 17 | 17 |
| Fucci-Dead_2 | 84 | 50 | 34 | 34/0 | 34/0 | 0 | 0 | 17 | 17 |

The only analysis-relevant exclusion is `Fucci-Tri_3 / WT_Fucci-Tri_3_20260914 / field150_res1536768_lFoV_zstack100to300_vol50`: PMT A/B `10/10` (expected `10/10`), 920 start/stop `0/0` (inactive), 1050 start/stop `50/50` (expected `60/60`).

The previous 11 exclusions were re-adjudicated after `_vol10` removal: **YES**. The remaining genuine analysis exclusion is the single 1050-power mismatch above.

## BASELINE-MANIFEST COMPARISON

Existing manifests predate the later sessions and the QC gate. Counts below compare their selected session lists with the fresh eligible catalog; additions are newly available valid sessions, not silent losses.

| mouse | laser | previous | current | removed by QC | added since baseline |
|---|---:|---:|---:|---|---:|
| Fucci-Tri_1 | 1050 | 33 | 38 | none | 5 |
| Fucci-Tri_1 | 920 | 0 | 29 | none | 29 |
| Fucci-Tri_3 | 1050 | 7 | 6 | 20260914 (1050 mismatch) | 0 |
| Fucci-Tri_3 | 920 | 1 | 7 | none | 6 |
| Fucci-Dead_1 | 1050 | 14 | 17 | none | 3 |
| Fucci-Dead_1 | 920 | 10 | 17 | none | 7 |
| Fucci-Dead_2 | 1050 | 14 | 17 | none | 3 |
| Fucci-Dead_2 | 920 | 12 | 17 | none | 5 |

## MASTER-PIPELINE BEHAVIOR

- Project manifests select only rows with `analysis_included=true` and `analysis_eligible=true`.
- A stale/manual manifest containing an ineligible row is rejected by the existing manifest validation guard.
- Pre-run outputs include `acquisition_settings_qc.csv`, `acquisition_settings_qc.png`, and `acquisition_settings_qc.json`.

## PLOTTING

- Four readable parameter panels (PMT A, PMT B, 920 Pockels, 1050 Pockels): **YES**
- `_vol10` absent: **YES**
- Failed setting marked at actual value: **YES**
- Inactive laser falsely marked: **NO**
- A/B expected series handled correctly: **YES**
- X labels readable/unclipped: **YES**
- Legend readable/unclipped: **YES**

## HARD CONSTRAINTS

- Matcher behavior changed: **NO**
- Track IDs changed: **NO**
- Phase D rescue changed: **NO**
- ECLIPSE/intensity calculations changed: **NO**

