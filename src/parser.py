from __future__ import annotations

import argparse
from PIL import Image
from .lsb import extract_lsb, LSBConfig
from .perturb import jpeg_compress, resize, rotate, blur


def main():
    ap = argparse.ArgumentParser(description="LSB parser (extractor)")
    ap.add_argument('--input', required=True, help='input stego image')
    ap.add_argument('--key', required=True, help='secret key')
    ap.add_argument('--mode', default='replace', choices=['replace', 'match'])
    ap.add_argument('--channel', default='ALL', choices=['ALL', 'R', 'G', 'B'])
    ap.add_argument('--jpeg', type=int, default=None, help='simulate JPEG quality before parsing')
    ap.add_argument('--resize', type=float, default=None, help='simulate resize scale before parsing')
    ap.add_argument('--rotate', type=float, default=None, help='simulate rotation degrees before parsing')
    ap.add_argument('--blur', type=float, default=None, help='simulate gaussian blur radius before parsing')
    args = ap.parse_args()

    img = Image.open(args.input).convert('RGB')
    if args.jpeg is not None:
        img = jpeg_compress(img, args.jpeg)
    if args.resize is not None:
        img = resize(img, args.resize)
    if args.rotate is not None:
        img = rotate(img, args.rotate)
    if args.blur is not None:
        img = blur(img, args.blur)

    cfg = LSBConfig(mode=args.mode, channel=args.channel)
    data = extract_lsb(img, args.key, cfg)
    try:
        text = data.decode('utf-8')
    except Exception:
        text = data.decode('utf-8', errors='ignore')
    print(text)


if __name__ == '__main__':
    main()
