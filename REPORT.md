Mosaic Lab Report

1. Method

Input images are processed through the following steps:

* Preprocessing: Convert image to uint8 RGB format (remove alpha channel, scale floats to 0–255) and center-crop so dimensions align with the grid size.
* Patching: Divide the image into p × p cells using vectorized reshaping (H, W, 3) to (gy, p, gx, p, 3), avoiding Python loops.
* Cell Color: Calculate the average RGB color for each cell across patch axes.
* Classification: Map cell colors to the nearest match in a standard 6-color palette (red, green, blue, yellow, black, white) using Euclidean distance.
* Tiling: Apply one of three rendering styles:
* solid: Fill each cell with its mapped palette color for a posterized effect.
* smooth: Fill each cell with its actual average color, preserving high similarity.
* pattern: Overlay one of 7 procedural patterns (gradients, checkerboard, stripes, radial) tinted by the cell's mean color while maintaining brightness.


* Evaluation: Calculate MSE and SSIM relative to the cropped original image alongside total cell count.

The Gradio interface (app.py) provides a slider for grid sizes from 8 to 64 pixels and options for tile styles, updating metrics live with each change.

2. Metrics

Evaluated on a 350×350 RGB image (image.png). Execution times reflect the fastest run out of 3 full pipeline executions (including pattern generation for pattern mode).

Grid (px) | Style | Cells | MSE | SSIM | Time (ms)
16 | solid | 21×21 | 5426.07 | 0.2165 | 4.38
16 | smooth | 21×21 | 969.38 | 0.4637 | 4.21
16 | pattern | 21×21 | 1727.86 | 0.3582 | 6.52
32 | solid | 10×10 | 6601.49 | 0.2014 | 3.45
32 | smooth | 10×10 | 1649.77 | 0.3884 | 3.62
32 | pattern | 10×10 | 2448.58 | 0.2968 | 6.78
64 | solid | 5×5 | 8423.27 | 0.1914 | 4.36
64 | smooth | 5×5 | 2652.99 | 0.3235 | 3.23
64 | pattern | 5×5 | 3563.75 | 0.2450 | 5.62

Vectorization Benchmark (10×10 grid, 32px cells, best of 5):

* Nested loops: 187.2 µs
* Reshape/transpose: 1.3 µs (~144× speedup with identical output)

3. Results

* Smooth mode achieves the highest fidelity across all grid sizes (lowest MSE, highest SSIM) because it preserves exact cell color averages, acting like a direct pixelation.
* Solid mode sacrifices accuracy for visual style, yielding higher error rates due to the strict 6-color limit.
* Pattern mode balances detail and aesthetics, retaining structural cues through brightness normalization at a minimal computational cost.
* Larger cell sizes reduce image quality, leading to steady decreases in SSIM across all styles.
* Processing speeds easily support real-time interaction, finishing under 7 ms per frame on 350×350 inputs to allow instant UI feedback in Gradio.

Vectorized operations keep execution fast enough for real-time adjustments, while the three rendering options provide flexibility between visual precision and stylized output.