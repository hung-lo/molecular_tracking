# Phase D1 Local Segmentation Rescue Pre-Pilot Hardening Report (2026-09-16)

baseline: `2e53dc4`
ending commit: `78873e8`
focused tests: `33 passed`
full suite: `334 passed, 2 skipped`

## TRUSTED SYNTHETIC SET

- evidence source: explicit track summaries, with canonical serialized edge and cycle-check fallback
- required fields/evidence: consensus source, cycle status, transform reliability, adjacent-edge stability, bounded distance, bounded ambiguity
- trusted n: `5002` on the available canonical run
- excluded n: `1657`
- missing-evidence n: `0` on the available canonical run; missing evidence now fails closed
- hard-coded trust claims remaining: **NO**
- each trusted case records `synthetic_trust_status`, `synthetic_trust_reasons`, `synthetic_trust_evidence_source`, and `synthetic_trust_fields_verified`

## REVIEW SAMPLING

- selection population = full benchmark: **YES**
- category-aware: **YES**
- implementation: bounded per-category reservoirs; no first-64 truncation

## CELLPOSE CACHE/PROVENANCE

- model cache keyed by device/model: **YES**
- reported device matches actual model: **YES**
- cellpose version: unavailable in this environment
- torch version: unavailable in this environment
- CUDA: unavailable in this environment

## ENDPOINT PARSING

- NaN alias fallback tested: **YES**
- missing artifact fail-closed: **YES**
- zero-valid-row fail-closed: **YES**
- classification source path + SHA256 recorded: **YES**

## SCIENTIFIC PILOT

- run: **NO**
- reason: Cellpose, Torch, and CUDA are unavailable here; the threshold backend remains an explicit test/negative-control backend only.
- n attempted: `N/A`
- identity_correct_rate: `N/A`
- wrong_neighbor_identity_rate: `N/A`
- median Dice: `N/A`
- median IoU: `N/A`
- median centroid error: `N/A`

## HARD CONSTRAINTS

- canonical masks changed: **NO**
- canonical tracks changed: **NO**
- primary extraction changed: **NO**
- state/intensity used for identity: **NO**
- production acceptance thresholds defined: **NO**

## UNRELATED ACQUISITION QC

- intentional independent feature: **YES**
- real catalog validation completed: **YES — read-only dry run**
- catalog rows scanned: `335`; discovery errors: `0`
- configured Fucci QC exclusions: `11` rows, each with an explicit PMT/Pockels mismatch reason
- unconfigured Fucci IDs (`Fucci-Tri_2`, `Fucci-Tri_4`): fail closed with an explicit missing-configuration reason
- sessions excluded unexpectedly: **NO silent exclusions; baseline comparison remains pending**

Phase D1 remains validation-only until the actual `cpsam_v2` GPU pilot is complete and its category-aware review panels are manually inspected.
