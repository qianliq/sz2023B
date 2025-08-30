from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Iterable
import numpy as np
from PIL import Image
import hashlib


@dataclass
class LSBConfig:
    mode: str = "replace"  # "replace" or "match" (±1 LSB matching)
    channel: str = "ALL"  # R|G|B|ALL
    seed: str | None = None  # if set, overrides key for pixel shuffle


def _rng_from_key(key: str) -> np.random.Generator:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    return np.random.default_rng(seed)


def _select_channels(arr: np.ndarray, channel: str) -> Iterable[int]:
    if arr.ndim == 2:
        return [0]
    if channel.upper() == "R":
        return [0]
    if channel.upper() == "G":
        return [1]
    if channel.upper() == "B":
        return [2]
    return [0, 1, 2]


def _img_to_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _array_to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _bits_from_bytes(data: bytes) -> np.ndarray:
    a = np.frombuffer(data, dtype=np.uint8)
    bits = ((a[:, None] >> np.arange(8)[::-1]) & 1).astype(np.uint8)
    return bits.reshape(-1)


def _bytes_from_bits(bits: np.ndarray) -> bytes:
    nbytes = (len(bits) + 7) // 8
    pad = nbytes * 8 - len(bits)
    if pad:
        bits = np.pad(bits, (0, pad), constant_values=0)
    b = np.packbits(bits.reshape(-1, 8), axis=1, bitorder='big')
    return b.tobytes()


def _capacity(arr: np.ndarray, channels: Iterable[int]) -> int:
    h, w = arr.shape[:2]
    return h * w * len(list(channels))


def _permute_indices(n: int, key: str) -> np.ndarray:
    rng = _rng_from_key(key)
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p.astype(np.float64); q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = (p > 0)
    return float(np.sum(p[m] * (np.log(p[m] + eps) - np.log(q[m] + eps))))


def _lsb_hist(arr: np.ndarray, channels: Iterable[int]) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    for c in channels:
        plane = arr[..., c].reshape(-1)
        ones = int(np.count_nonzero(plane & 1))
        zeros = plane.size - ones
        stats[f"ch{c}_ones"] = ones
        stats[f"ch{c}_zeros"] = zeros
        stats[f"ch{c}_ones_ratio"] = ones / max(1, plane.size)
    return stats


def embed_lsb(img: Image.Image, payload: bytes, key: str, cfg: LSBConfig | None = None) -> Tuple[Image.Image, Dict]:
    cfg = cfg or LSBConfig()
    arr = _img_to_array(img)
    chs = _select_channels(arr, cfg.channel)
    cap = _capacity(arr, chs)

    # Header: 32-bit big-endian length
    header = len(payload).to_bytes(4, 'big')
    bits = _bits_from_bytes(header + payload)
    if len(bits) > cap:
        raise ValueError(f"Payload too large: need {len(bits)} bits, capacity {cap}")

    h, w = arr.shape[:2]
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1).reshape(-1, 2)
    channels = np.repeat(np.array(chs, dtype=np.int32), h * w)
    coords = np.tile(coords, (len(chs), 1))
    total = coords.shape[0]
    order = _permute_indices(total, cfg.seed or key)

    carrier_vals = arr[coords[:, 0], coords[:, 1], channels]
    before_hist, _ = np.histogram(carrier_vals, bins=256, range=(0, 256))

    # Select positions to embed
    sel = order[: len(bits)]
    vals = carrier_vals.copy()
    if cfg.mode == "match":
        # LSB matching: if LSB equals target bit, keep; else ±1 with equal prob
        cur = vals[sel]
        need = bits
        mask = ((cur & 1) != need)
        rng = _rng_from_key(key + ":match")
        delta = rng.integers(0, 2, size=mask.sum(), endpoint=False)
        delta = np.where(delta == 0, -1, 1)
        new = cur.copy()
        new[mask] = np.clip(cur[mask].astype(np.int16) + delta.astype(np.int16), 0, 255).astype(np.uint8)
        vals[sel] = new
    else:
        # LSB replacement
        vals_bits = vals[sel] & 0xFE
        vals[sel] = vals_bits | bits

    # Write back
    stego = arr.copy()
    stego[coords[:, 0], coords[:, 1], channels] = vals

    after_hist, _ = np.histogram(stego[coords[:, 0], coords[:, 1], channels], bins=256, range=(0, 256))
    info = {
        "capacity_bits": cap,
        "used_bits": int(len(bits)),
        "payload_len": len(payload),
        "mode": cfg.mode,
        "hist_kl": _kl_divergence(before_hist, after_hist),
        "lsb_before": _lsb_hist(arr, chs),
        "lsb_after": _lsb_hist(stego, chs),
    }
    return _array_to_img(stego), info


def extract_lsb(img: Image.Image, key: str, cfg: LSBConfig | None = None) -> bytes:
    cfg = cfg or LSBConfig()
    arr = _img_to_array(img)
    chs = _select_channels(arr, cfg.channel)

    h, w = arr.shape[:2]
    coords = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1).reshape(-1, 2)
    channels = np.repeat(np.array(chs, dtype=np.int32), h * w)
    coords = np.tile(coords, (len(chs), 1))
    total = coords.shape[0]
    order = _permute_indices(total, cfg.seed or key)

    vals = arr[coords[:, 0], coords[:, 1], channels]
    # First 32 bits = length
    hdr_sel = order[:32]
    hdr_bits = vals[hdr_sel] & 1
    nbytes = int.from_bytes(_bytes_from_bits(hdr_bits)[:4], 'big')
    bitlen = (nbytes + 4) * 8
    sel = order[:bitlen]
    bits = vals[sel] & 1
    data = _bytes_from_bits(bits)
    return data[4:4 + nbytes]
