from __future__ import annotations

import numpy as np
from PIL import Image
from typing import Dict, Tuple
from .lsb import _img_to_array, _select_channels


def lsb_plane(img: Image.Image, channel: str = "ALL") -> np.ndarray:
    arr = _img_to_array(img)
    chs = _select_channels(arr, channel)
    if len(chs) == 1:
        plane = (arr[..., chs[0]] & 1).astype(np.uint8)
    else:
        plane = (arr[..., chs] & 1).mean(axis=2)
    return plane


def hist_256(img: Image.Image, channel: str = "ALL") -> Dict[str, np.ndarray]:
    arr = _img_to_array(img)
    chs = _select_channels(arr, channel)
    out: Dict[str, np.ndarray] = {}
    for c in chs:
        h, _ = np.histogram(arr[..., c].reshape(-1), bins=256, range=(0, 256))
        out[f"ch{c}"] = h
    return out


def chi_square_lsb(img: Image.Image, channel: str = "ALL") -> float:
    # Westfeld chi-square attack statistic on pairs (2i,2i+1)
    arr = _img_to_array(img)
    chs = _select_channels(arr, channel)
    xs = []
    for c in chs:
        h, _ = np.histogram(arr[..., c].reshape(-1), bins=256, range=(0, 256))
        o = 0.0
        for i in range(0, 256, 2):
            e = (h[i] + h[i + 1]) / 2.0
            if e > 0:
                o += (h[i] - e) ** 2 / e + (h[i + 1] - e) ** 2 / e
        xs.append(o)
    return float(np.mean(xs))


def sample_pair_estimate(img: Image.Image, channel: str = "ALL") -> float:
    # A rough sample pair analysis estimate of embedding rate (LSB replacement)
    arr = _img_to_array(img)
    chs = _select_channels(arr, channel)
    # Use simple neighbor pairs horizontally
    ests = []
    for c in chs:
        plane = arr[..., c].astype(np.int16)
        a = plane[:, :-1].reshape(-1)
        b = plane[:, 1:].reshape(-1)
        # pairs categorized by parity and difference
        e = np.count_nonzero((a % 2 == 0) & (b == a + 1))
        f = np.count_nonzero((a % 2 == 1) & (b == a - 1))
        g = np.count_nonzero(a == b)
        # estimator (simplified) p ≈ (e + f) / (e + f + g + 1e-9)
        p = (e + f) / max(1, (e + f + g))
        ests.append(p)
    return float(np.mean(ests))
