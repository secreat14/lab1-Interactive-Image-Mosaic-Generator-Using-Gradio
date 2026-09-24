"""
Mosaic Generator (Gradio)
========================
Cut an image into grid cells, classify each cell by color, and swap in a matching tile.

Run locally: gradio app.py   (not python app.py, the gradio command hot-reloads)
Deploy: Hugging Face Spaces, not the 3-day temporary link. requirements.txt:
    gradio==6.28.0
    numpy==2.5.1
    pillow==12.3.0
    scikit-image==0.25.2
"""

import gradio as gr
import numpy as np

try:
    from skimage.metrics import structural_similarity
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


# 6 个颜色的调色板
PALETTE = np.array([
    [255, 0, 0],      # 红
    [0, 255, 0],      # 绿
    [0, 0, 255],      # 蓝
    [255, 255, 0],    # 黄
    [0, 0, 0],        # 黑
    [255, 255, 255],  # 白
], dtype=np.uint8)


def to_uint8_rgb(img):
    # 转成 uint8 的 RGB: 透明通道扔掉, 小数转成整数
    img = np.asarray(img)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
    return np.ascontiguousarray(img)


def crop_to_grid(img, p):
    # 从中间裁一刀, 让高和宽都变成 p 的整数倍
    h, w = img.shape[:2]
    h2 = (h // p) * p
    w2 = (w // p) * p
    y0 = (h - h2) // 2
    x0 = (w - w2) // 2
    return np.ascontiguousarray(img[y0:y0 + h2, x0:x0 + w2])


def preprocess(img, grid_size):
    # 转格式, 再裁成格子的整数倍
    img = to_uint8_rgb(img)
    return crop_to_grid(img, grid_size)

def cell_means(img, p):
    # 向量化算每块的平均色, 不用循环
    # 把 (高, 宽, 3) 看成 (块行数, 块内行, 块列数, 块内列, 3), 对块内求平均
    h, w = img.shape[:2]
    big = img.reshape(h // p, p, w // p, p, 3)
    return big.mean(axis=(1, 3))




def classify_cells(cells):
    # 每个格子找离它最近的调色板颜色, 广播一次算完
    diff = cells[..., None, :].astype(np.float64) - PALETTE.astype(np.float64)
    dists = np.linalg.norm(diff, axis=-1)
    return np.argmin(dists, axis=-1)


def make_pattern_bank(t=32):
    # 现场生成 7 种 tile 图案
    solid = np.ones((t, t))                                    # 纯色
    h_grad = np.tile(np.linspace(0.5, 1.0, t), (t, 1))         # 横向渐变
    v_grad = np.tile(np.linspace(0.5, 1.0, t)[:, None], (1, t))  # 纵向渐变

    yy, xx = np.mgrid[0:t, 0:t].astype(float)
    checker = ((xx // (t // 4) + yy // (t // 4)) % 2) * 0.5 + 0.5   # 棋盘格
    v_stripes = 0.75 + 0.25 * np.sin(2 * np.pi * xx / t * 3)        # 竖条纹
    d_stripes = 0.75 + 0.25 * np.sin(2 * np.pi * (xx + yy) / t * 2) # 斜条纹

    cx = (t - 1) / 2
    cy = (t - 1) / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (np.sqrt(2) * (t - 1) / 2)
    radial = 1.0 - 0.5 * r                                        # 放射渐变

    # 每个图案除以自己的均值, 平均亮度变成 1, 染色后跟原格子差不多亮
    pats = [solid, h_grad, v_grad, checker, v_stripes, d_stripes, radial]
    bank = []
    for p in pats:
        bank.append(np.clip(p / p.mean(), 0, 1.5))
    return bank


def build_mosaic(img, grid_size=32, tile_style="solid"):
    # 主函数: 输入原图, 输出 (马赛克, 裁剪后的原图)
    #   'solid'   -> 每个格子填调色板颜色 (6 色海报风)
    #   'smooth'  -> 每个格子填自己的平均色 (最像原图)
    #   'pattern' -> 每个格子贴图案 tile, 再按平均色染色 (创意)
    img = preprocess(img, grid_size)
    cells = cell_means(img, grid_size)
    gy = cells.shape[0]
    gx = cells.shape[1]
    p = grid_size

    if tile_style == "smooth":
        mosaic = np.repeat(np.repeat(cells, p, axis=0), p, axis=1)

    elif tile_style == "solid":
        cat = classify_cells(cells)
        colors = PALETTE[cat]
        mosaic = np.repeat(np.repeat(colors, p, axis=0), p, axis=1)

    elif tile_style == "pattern":
        bank = np.stack(make_pattern_bank(p))
        cat = classify_cells(cells)
        sel = bank[cat % len(bank)]
        tinted = sel[..., None] * cells[:, :, None, None, :]
        mosaic = tinted.transpose(0, 2, 1, 3, 4).reshape(gy * p, gx * p, 3)

    else:
        raise ValueError("tile_style must be solid / smooth / pattern")

    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)
    return mosaic, img


def mse(a, b):
    # MSE: 逐像素作差, 平方, 再平均。越小越好
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff ** 2))


def ssim_index(a, b):
    # SSIM: 综合看亮度、对比度、结构, 越接近 1 越好
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


# 界面上的三种风格说明
TILE_STYLES = ["solid", "smooth", "pattern"]
STYLE_HELP = (
    "**solid** - each cell becomes a palette color, 6-color poster look.\n\n"
    "**smooth** - each cell becomes its own mean color, closest to the original.\n\n"
    "**pattern** - each cell gets a pattern tile, tinted by its mean color."
)


def run_mosaic(image, grid_size, tile_style):
    # Gradio 点按钮后跑的函数
    if image is None:
        raise gr.Error("Please upload an image first")

    grid_size = int(grid_size)
    mosaic, base = build_mosaic(image, grid_size, tile_style)

    gy = base.shape[0] // grid_size
    gx = base.shape[1] // grid_size

    info = "MSE: " + str(round(mse(base, mosaic), 2))
    info = info + "   |   Cells: " + str(gy) + "x" + str(gx)
    if _HAS_SSIM:
        info = info + "   |   SSIM: " + str(round(ssim_index(base, mosaic), 4))

    return mosaic, info


with gr.Blocks(title="Interactive Image Mosaic Generator") as demo:
    gr.Markdown(
        "# Mosaic Generator\n"
        "Upload an image, pick a grid size and tile style, get a mosaic in real time.\n\n" + STYLE_HELP
    )

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="numpy", label="Input image")
            grid = gr.Slider(8, 64, value=32, step=8, label="Grid size (pixels per cell)")
            style = gr.Radio(TILE_STYLES, value="solid", label="Tile style")
            btn = gr.Button("Generate mosaic", variant="primary")

        with gr.Column():
            img_out = gr.Image(label="Mosaic result")
            metrics = gr.Textbox(label="Similarity metrics", interactive=False)

    btn.click(run_mosaic, inputs=[img_in, grid, style], outputs=[img_out, metrics])

    # 作业 tip: 界面下方放一键示例, 点一下 image.png 就能直接测, 不用手动上传
    gr.Examples(examples=["image.png"], inputs=img_in)


if __name__ == "__main__":
    demo.launch()
