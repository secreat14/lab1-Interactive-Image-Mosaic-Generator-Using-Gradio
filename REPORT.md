# Mosaic Lab Report

## 1. Method

Given an input image, the pipeline is:

1. **Preprocess** — convert to `uint8` RGB (drop the alpha channel, scale floats to 0–255), then center-crop so height and width are multiples of the grid size.
2. **Patching** — split the image into `p × p` cells with a vectorized reshape
   `(H, W, 3) → (gy, p, gx, p, 3)`; no Python loops in the hot path.
3. **Cell color** — mean color of each cell, computed over the patch axes.
4. **Classification** — each cell is assigned the nearest color of a fixed 6-color
   palette (red, green, blue, yellow, black, white) using broadcasted Euclidean distance.
5. **Tiling** — three styles:
   - `solid`: fill the cell with its palette color (6-color poster look);
   - `smooth`: fill the cell with its own mean color (closest to the original);
   - `pattern`: paste one of 7 generated pattern tiles (gradients, checkerboard,
     stripes, radial), tinted by the cell's mean color while preserving luminance.
6. **Evaluation** — MSE and SSIM against the cropped original, plus cell count,
   shown live in the Gradio app.

The interactive demo (`app.py`) exposes a grid-size slider (8–64 px) and the three
tile styles, and reports MSE / SSIM / cell count on every run.

## 2. Metrics

Test image: 350×350 RGB (`image.png`). Time is best of 3 runs of the full
`build_mosaic` pipeline (includes pattern-bank generation for `pattern`).

| Grid (px) | Style   | Cells  | MSE     | SSIM   | Time (ms) |
|-----------|---------|--------|---------|--------|-----------|
| 16        | solid   | 21×21  | 5426.07 | 0.2165 | 4.38      |
| 16        | smooth  | 21×21  | 969.38  | 0.4637 | 4.21      |
| 16        | pattern | 21×21  | 1727.86 | 0.3582 | 6.52      |
| 32        | solid   | 10×10  | 6601.49 | 0.2014 | 3.45      |
| 32        | smooth  | 10×10  | 1649.77 | 0.3884 | 3.62      |
| 32        | pattern | 10×10  | 2448.58 | 0.2968 | 6.78      |
| 64        | solid   | 5×5    | 8423.27 | 0.1914 | 4.36      |
| 64        | smooth  | 5×5    | 2652.99 | 0.3235 | 3.23      |
| 64        | pattern | 5×5    | 3563.75 | 0.2450 | 5.62      |

Vectorization check (10×10 grid of 32 px cells, best of 5): nested-loop patch
extraction 187.2 µs vs. vectorized reshape/transpose 1.3 µs — **~144× faster**,
bit-identical output.

## 3. Results

- **`smooth` wins on fidelity at every grid size** (lowest MSE, highest SSIM), as
  expected — each cell keeps its true mean color, so the mosaic is a blocky
  downsample of the original.
- **`solid` trades fidelity for style**: restricting every cell to 6 palette
  colors gives the poster aesthetic but the worst MSE/SSIM.
- **`pattern` sits in between**: the luminance-normalized tinted tiles preserve
  more structure than flat palette fills, at a small cost in build time
  (pattern-bank generation).
- **Larger cells → worse fidelity**: with fewer, coarser cells both metrics degrade
  monotonically (e.g. smooth SSIM 0.4637 → 0.3235 from 16 px to 64 px cells).
- **Performance is comfortably real-time**: the whole pipeline runs in under
  7 ms on a 350×350 image, so the Gradio demo regenerates instantly when the
  slider or style changes.

In short: vectorized NumPy keeps the pipeline fast enough for live interaction,
and the three tile styles span the fidelity–style trade-off — `smooth` for
likeness, `solid` for the poster look, `pattern` as the creative middle ground.
