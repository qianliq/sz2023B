from __future__ import annotations

from typing import Tuple
from PIL import Image, ImageFilter
import io


def jpeg_compress(img: Image.Image, quality: int = 85) -> Image.Image:
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def resize(img: Image.Image, scale: float = 0.5) -> Image.Image:
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def rotate(img: Image.Image, degrees: float = 2.0) -> Image.Image:
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False)


def blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))
