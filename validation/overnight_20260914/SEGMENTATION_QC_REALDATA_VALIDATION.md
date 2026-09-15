# Segmentation-QC real-data validation

- Validation date: 2026-09-14/15
- Starting commit: `de3cdefbbcffb00ee71f544d56065202e3269424`
- Dataset: `Fucci-Dead_1`, 1050 nm, first four sessions
- Canonical run: `1050_20260819_to_20260824_4s_first4_graph_affine_balanced`
- Validation result: **PASS**
- Segmentation-QC real-data validation passed: **YES**

The extraction stage was reproduced from the canonical matching directory in a new persistent validation directory. Canonical matching and extraction outputs were not overwritten.

| Filter step | Before (`04d522f`) | After (`de3cdef`) |
|---|---:|---:|
| all_tracks | 6,983 | 6,983 |
| at_least_two_sessions | 5,054 | 5,054 |
| complete_required_sessions | 3,251 | 3,251 |
| zero_hit_complete | 3,251 | 3,251 |
| cycle_qc | 3,216 | 3,216 |
| segmentation_qc | 0 | 3,216 |
| primary_final | 0 | 3,216 |
| one_internal_gap | 169 | 169 |

Required field checks across all 6,983 rows of `matched_track_qc_summary.csv`:

- `segmentation_qc_status = not_configured`: 6,983/6,983
- `segmentation_qc_pass_fraction = NaN`: 6,983/6,983
- `segmentation_qc_pass_all_required_days = NA/NaN`: 6,983/6,983
- `segmentation_failure = False`: 6,983/6,983
- Run-level `segmentation_qc_status = not_configured_bypassed`
- Filter row `step_status = bypassed_not_configured`

The exact original extraction configuration requested only the `graph` policy. Therefore the named high/balanced sensitivity files have zero rows by policy selection, not because segmentation-QC removed them. The requested graph primary output contains 3,216 tracks after the bypass.

Matching identity was held fixed by consuming the existing canonical matching artifacts. Selected provenance hashes:

| Artifact | SHA-256 |
|---|---|
| selected session manifest | `3360a8d46ba0798ee0e9d1e932f14ee18bae3f54c1c8526a241cec1051d0f239` |
| matching run log | `19deaeeaa4e061f03d2f968d35f7b83ff690c6466df0c80ecfac07b0c8ecba5e` |
| graph pairwise matches | `e3e38022c967bc3cdfe65f41452228ba431d8ff1c83772e7fcdc3e00090dec5e` |
| graph track edges | `c90c36fef4bc8482972219582246ad49a8aef7cd38430b9c4fd9b7d1ae9c01b2` |
| graph tracks | `51e42bbbd399b5c6a9a7a799fab65b717b3056c9c5035a4b1a4051d24f9f4d00` |

