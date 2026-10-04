# Changelog

## 1.0.0 (October 2026)

- **Web app** rewritten (`ascal web`, online at https://lumaria.allandestars.com/ascal/): job queue with one process
  per calibration, zoomable layered viewer, tables and interactive ECharts diagnostics, exports; Docker deployment
  and reverse-tunnel recipe in `deploy/`.

Qualified on 23 all-sky systems (DFN, FRIPON, MMT, ESO Paranal, Liverpool Telescope, KLCAM, Allsky network
stations and the Lumaria camera), from 0.3 to 77 Mpx. Single-frame calibration is an incremental cascade of
hypotheses with a time budget.

- **Cascade** (`ascal.fast`): sky disc as detected, then sensor-inscribed and circumscribed circles, then Hough
  circles; both parities; radial priors (equisolid-, stereographic-, equidistant-like); wider detection kernels as
  a last resort. Stops at the first calibration that passes the gate. `--max-time` (default 40 s).
- **Pose search** on a distance map of the detections: about 1 s per hypothesis for both parities (was 8–10 s).
- **Gate** scaled to the frame: tight fit, and either 15 % of the stars expected down to the frame's own limiting
  magnitude or a pose that beats every rival by a factor of 2; the fixed 80-pair gate of 0.x rejected small images
  and accepted false poses on large ones.
- **Parity**: mirrored images (FITS, some cameras) are handled; `CameraModel.mirror` keeps the model in the pixels
  of the image as given.
- **Detection**: Gaussian pre-smoothing of undersampled stars (FWHM < 2.2 px); threshold from the MAD of the
  background-subtracted frame (the box rms included the Moon and vignetting gradients); background box and all
  matching radii scaled with the plate scale; 3 × 3 tiles in parallel above 16 Mpx.
- **Refraction**: catalogue positions are compared with apparent altitudes (Saemundsson 1986, pressure from the
  site elevation); `--no-refraction` restores 0.x.
- **Inputs**: FITS (time and exposure from the header) and camera raw files (optional `rawpy`).
- **CLI**: `--max-time`, `--parity`, `--disc`, `--no-refraction`, `--tiles auto`; the output says which hypothesis
  was accepted. Options in the API through `ascal.config.options(...)`.
- **Report by default**: `ascal calibrate` writes `<image>_ascal/` (or `--report DIR`; `--no-report` to skip) with
  the calibration, summary, pairs and figures, now including `panel.png`, a 2 x 2 summary (frame; matched stars with
  altitude circles, N-S/E-W lines and constellation figures; altitude vs radius; residual vs altitude), also shown
  by the web demo and available as `plots.calibration_panel`. Constellation figures from d3-celestial (BSD 3-Clause).
- Multi-frame calibration: the frame with most detections goes through the cascade, then all frames are refined
  together.

## 0.3.0

Detection kernel from the measured star FWHM (`--fwhm auto`), retries with other widths.
