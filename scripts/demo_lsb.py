from __future__ import annotations

import os
from PIL import Image
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lsb import embed_lsb, extract_lsb, LSBConfig
from src.viz import plot_histograms, plot_lsb_planes
from src.analysis import chi_square_lsb, sample_pair_estimate


def run():
    inp = 'data/B题-附件1.jpg'
    out_dir = 'out_lsb'
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, 'stego.png')
    fig_hist = os.path.join(out_dir, 'hist.png')
    fig_plane = os.path.join(out_dir, 'lsb_plane.png')

    key = 'sz2023b-key'
    msg = '深圳杯2023B-数字版权演示: LSB + 样本对分析 + ±1匹配'

    img = Image.open(inp).convert('RGB')
    cfg = LSBConfig(mode='match')
    stego, info = embed_lsb(img, msg.encode('utf-8'), key, cfg)
    stego.save(out_img)

    # Visualize
    plot_histograms(img, stego, fig_hist)
    plot_lsb_planes(img, stego, fig_plane)

    # Analysis
    chi_b = chi_square_lsb(img)
    chi_a = chi_square_lsb(stego)
    spa_a = sample_pair_estimate(stego)

    # Extract
    rec = extract_lsb(stego, key, cfg)

    print('KL(hist):', info['hist_kl'])
    print('Chi-square before/after:', chi_b, chi_a)
    print('SPA estimate after:', spa_a)
    print('Recovered:', rec.decode('utf-8', errors='ignore'))


if __name__ == '__main__':
    run()
