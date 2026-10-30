# Samples

Provenance for every sample card in this directory.

## card-01

Kevin Masterson, the owner's own clip, recorded 2026-10-29, all rights reserved.

`./demo` needs no path argument — it already points at
`samples/card-01/CLIP_0001.mp4` (its sha256 manifest is
`samples/card-01/manifest.json`). That file is the owner's own solo clip,
recorded and cleared by him for this repo, transcoded by
`scripts/make-sample` to 720p (the vertical source is pillarboxed).

Swapping it needs no code change: `scripts/make-sample <clip>` transcodes
*any* clip into `samples/card-01/` (720p, 25 MB or less, a fresh
`manifest.json` with matching sha256 values, and the provenance line above
rewritten — the rest of this section is kept), and `./demo` picks up
whatever is there next time it runs. `scripts/make-stand-in-sample`
generates a synthetic stand-in (espeak-ng speech over an ffmpeg lavfi test
pattern, no real footage) if you want to run the pipeline on something else.
