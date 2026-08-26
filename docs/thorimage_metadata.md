# ThorImageLS metadata rules

`core/thorimage_xml.py` reads geometry and acquisition state from `Experiment.xml`.
XY spacing comes from `pixelWidthUM` and `pixelHeightUM`, cross-checked against physical
width/height divided by pixel counts. Z geometry comes from `ZStage`; inactive RemoteFocus
metadata and legacy `pixelSizeUM` are not used.

Pockels node 1 maps to 920 nm and node 2 maps to 1050 nm. Both ramp endpoints are retained.
Detector A is green and detector B is red. The live acquisition path is always the discovered
filesystem path; the XML `Name.path` value is informational provenance only.

Canonical quantitative acquisitions are complete 41-plane plus one-flyback, 5 µm, 50-volume,
2100-frame stacks. `_vol10` acquisitions are alignment-only. Missing 920 data are valid.
