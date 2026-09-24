# Interactive Image Mosaic Generator

Cut an image into grid cells, classify each cell by color, and replace it with a
matching tile — with a live Gradio demo.

## Files

| File              | What it is                                              |
|-------------------|---------------------------------------------------------|
| `app.py`          | Standalone Gradio app (upload image → mosaic + metrics) |
| `lab1.ipynb`      | Lab notebook: step-by-step build, timings, experiments  |
| `image.png`       | Sample image, also wired as a one-click Gradio example  |
| `requirements.txt`| Pinned dependencies                                     |
| `REPORT.md`       | One-to-two-page report: method, metrics, results         |

## Quickstart

```bash
pip install -r requirements.txt
gradio app.py
```

Pick a grid size (8–64 px per cell) and a tile style (`solid` / `smooth` / `pattern`);
the app shows the mosaic plus MSE, SSIM, and cell count.

## Live demo

https://huggingface.co/spaces/<your-username>/mosaic-lab
*(replace with your Space URL after deploying — see REPORT.md for the numbers behind it)*

## Report

Method, performance metrics, and results: [REPORT.md](REPORT.md)
