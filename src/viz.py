from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Dict
from .analysis import hist_256, lsb_plane


def plot_histograms(img_before: Image.Image, img_after: Image.Image, out_path: str) -> None:
    hb = hist_256(img_before)
    ha = hist_256(img_after)
    ch_keys = sorted(hb.keys())
    n = len(ch_keys)
    fig, axs = plt.subplots(n, 1, figsize=(8, 3 * n), tight_layout=True)
    if n == 1:
        axs = [axs]
    for i, k in enumerate(ch_keys):
        axs[i].plot(hb[k], label='before')
        axs[i].plot(ha[k], label='after')
        axs[i].set_title(f'Histogram {k}')
        axs[i].legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_lsb_planes(img_before: Image.Image, img_after: Image.Image, out_path: str) -> None:
    pb = lsb_plane(img_before)
    pa = lsb_plane(img_after)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
    axs[0].imshow(pb, cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('LSB plane before')
    axs[0].axis('off')
    axs[1].imshow(pa, cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('LSB plane after')
    axs[1].axis('off')
    fig.savefig(out_path)
    plt.close(fig)
