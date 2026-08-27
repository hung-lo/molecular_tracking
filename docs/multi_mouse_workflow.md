# Multi-mouse workflow

1. Copy `config/project.example.toml` to ignored `config/project.local.toml` and edit paths.
2. Audit without writing:

   ```bash
   python tools/build_data_catalog.py --project-config config/project.local.toml --dry-run
   ```

3. Write `_catalog` outputs, optionally adding `--strict` for CI validation.
4. Prepare one mouse's 1050 manifest:

   ```bash
   python tools/build_session_manifest.py --project-config config/project.local.toml --mouse-id Fucci-Tri_1 --laser-nm 1050
   ```

Until mask, red, and green derivative files exist, this writes
`session_manifest_plan.csv`, not a misleading analysis-ready manifest. Preprocessing and
physical migration are intentionally deferred. Use explicit dataset, manifest, match, and
output paths for legacy workflows.
The weekly registration notebook reads flat source TIFFs from `registered/`, stages its
compatibility outputs under `weekly_registered/.staging/`, and publishes the final flat
compatibility product into `weekly_registered/` only after validation. The weekly matcher
consumes that validated directory directly. Duplicate day/channel products are treated as
errors. The 920 compatibility wrapper writes project-mode outputs under
`<mouse>/longitudinal/920/runs/`.
