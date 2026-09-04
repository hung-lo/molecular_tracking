# Cross-laser 920-to-1050 ROI mapping

The isolated cross-laser workflow maps same-session native 920 ROI labels to
canonical 1050 ROI labels. It is independent of longitudinal tracking and
does not extract 920 fluorescence.

## Identity direction

- Fixed canonical source: 1050 red Cellpose mask.
- Primary moving source: 920 green Cellpose mask.
- Optional secondary evidence: 920 red mask at
  sessions/<date>/920/segmentation/mask_red.tif.
- Optional source-consistency relation: 920 red to 920 green.

The affine matcher is reused unchanged. Its transform direction is always
moving to fixed. For the primary relation this is 920 green to 1050 red.

## Run one paired session

    python tools/build_cross_laser_roi_map.py \
      --project-config config/project.local.toml \
      --mouse-id Fucci-Dead_1 \
      --session-id 20260819

To evaluate optional red evidence:

    python tools/build_cross_laser_roi_map.py \
      --project-config config/project.local.toml \
      --mouse-id Fucci-Dead_1 \
      --session-id 20260819 \
      --use-920-red-secondary

The tool reads only canonical analysis-included acquisitions from the catalog.
Sessions without an eligible partner are skipped in all-session mode; an
explicit request for one fails with an actionable error.

## Identity interpretation

Only a primary-green high assignment is recommended automatically. A
secondary-red high assignment is exported as a provisional rescue candidate
requiring review. A green/red source-consistency conflict blocks automatic red
rescue.

The workflow writes one coverage row for every 1050 ROI, including labels with
no candidates and labels outside the transformed observable 920 volume.

## Output location

Runs are written under:

    <derivatives>/<mouse>/cross_laser/920_to_1050/runs/<run_name>/

Key files include fixed and moving coverage, source-specific candidates and
high/balanced assignments, identity resolution, explicit transforms, numerical
QC, and a native-920 relabelled primary-high mask. Source masks are never
modified.
