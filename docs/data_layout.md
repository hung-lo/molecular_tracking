# Multi-mouse data layout

Raw ThorImage folders remain read-only under `raw_root/<raw_mouse_folder>/<session>/<acquisition>`.
No reorganization command renames, moves, copies, links, or rewrites them.

Generated catalogs and analysis products live under the separate `derivatives_root`:

```text
_catalog/
<mouse_id>/sessions/<YYYYMMDD>/<laser_nm>/{preprocessing,segmentation,qc}/
<mouse_id>/longitudinal/<laser_nm>/{manifests,registration,runs,qc}/
```

There is one FOV per mouse, so no FOV directory is introduced. Existing legacy averaged,
cropped, and aligned products remain where they are and are available only through explicit
legacy paths.
