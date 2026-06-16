# PUG 2026 slides

Quarto [revealjs](https://quarto.org/docs/presentations/revealjs/) deck for the
25-minute talk **"PolarisOpt: Model Calibration and Exploration Library."**

```bash
quarto render pug-2026.qmd        # -> pug-2026.html
quarto preview pug-2026.qmd       # live-reload while editing
```

The deck is **not executed** — all code blocks are plain fenced snippets, so
rendering needs only Quarto (no Python kernel, no polarisopt install).

The live demo is driven separately from
[`../notebooks/05_calibration.ipynb`](../notebooks/05_calibration.ipynb)
(kernel: `polaris_xover`). Budget ~8 minutes for it; the `run_study` cell
resumes the finished SampleStore in ~1 minute — narrate the master/slave loop
while it runs.

Files:

- `pug-2026.qmd` — the deck
- `custom.scss` — Argonne-flavored theme

Speaker notes (press `s` in the rendered deck) carry the demo cue and timing.
