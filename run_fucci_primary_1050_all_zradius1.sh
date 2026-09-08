#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/d/Codex_folder/molecular_tracking"
PROJECT_CONFIG="config/project.local.toml"

cd "$REPO"
source .venv/bin/activate

echo
echo "============================================================"
echo " Molecular tracking canonical rerun"
echo "============================================================"
echo

echo ">>> [1/4] Catalog dry run"
python tools/build_data_catalog.py \
  --project-config "$PROJECT_CONFIG" \
  --dry-run

echo
echo ">>> [2/4] Build strict catalog"
python tools/build_data_catalog.py \
  --project-config "$PROJECT_CONFIG" \
  --strict

run_mouse () {
    local mouse="$1"
    shift

    echo
    echo "============================================================"
    echo " Running: ${mouse} | 1050 nm"
    echo " Ranked ROI views: center z +/- 1 plane"
    echo "============================================================"

    python tools/build_session_manifest.py \
      --project-config "$PROJECT_CONFIG" \
      --mouse-id "$mouse" \
      --laser-nm 1050

    python core/run_daywise_master_pipeline.py \
      --project-config "$PROJECT_CONFIG" \
      --mouse-id "$mouse" \
      --laser-nm 1050 \
      --trajectory-min-sessions 2 \
      --ranked-roi-z-radius 1 \
      --overwrite \
      "$@"

    echo
    echo ">>> Completed ${mouse}"
}

# Fucci-Tri_1: first 20 sessions only
run_mouse "Fucci-Tri_1" \
  --sessions first:20

# Fucci-Dead controls: all available 1050 sessions
run_mouse "Fucci-Dead_1"
run_mouse "Fucci-Dead_2"

echo
echo "============================================================"
echo " ALL REQUESTED 1050-NM RUNS COMPLETED SUCCESSFULLY"
echo "============================================================"
echo "Completed:"
echo "  Fucci-Tri_1  : first 20 sessions"
echo "  Fucci-Dead_1 : all available 1050 sessions"
echo "  Fucci-Dead_2 : all available 1050 sessions"
echo
echo "Ranked single-ROI views:"
echo "  z radius = 1  (center plane +/- 1; 3 planes total)"
echo
echo "Not run:"
echo "  Fucci-Tri_2"
echo "  Fucci-Tri_3"
echo "  any 920-nm dataset"
