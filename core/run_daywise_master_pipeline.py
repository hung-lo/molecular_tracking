"""Compatibility wrapper for the daywise master ROI pipeline runner.

The full implementation currently lives in
``chatgpt_related/run_daywise_master_pipeline.py`` so we can keep the draft
workflow alongside the source notes. This module makes the runner available in
the repo's standard ``core/`` entry-point location without changing the actual
pipeline behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chatgpt_related.run_daywise_master_pipeline import (  # noqa: E402
    MasterPipelineConfig,
    main,
    parse_args,
    run_master_pipeline,
)

__all__ = [
    "MasterPipelineConfig",
    "main",
    "parse_args",
    "run_master_pipeline",
]


if __name__ == "__main__":
    main()
