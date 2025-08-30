from __future__ import annotations

import argparse
import json
import os
from PIL import Image
from .lsb import embed_lsb, LSBConfig
from .viz import plot_histograms, plot_lsb_planes


def main():
    ap = argparse.ArgumentParser(description="LSB encryptor (embedder)")
    ap.add_argument('--input', required=True, help='input image (cover)')
    ap.add_argument('--out', required=True, help='output stego image path')
    ap.add_argument('--key', required=True, help='secret key')
    ap.add_argument('--message', required=True, help='text to embed')
    ap.add_argument('--mode', default='replace', choices=['replace', 'match'], help='LSB mode')
    ap.add_argument('--channel', default='ALL', choices=['ALL', 'R', 'G', 'B'])
    ap.add_argument('--report', default=None, help='optional JSON report path')
    ap.add_argument('--figdir', default=None, help='optional directory to save figures')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.figdir:
        os.makedirs(args.figdir, exist_ok=True)

    img = Image.open(args.input).convert('RGB')
    cfg = LSBConfig(mode=args.mode, channel=args.channel)
    stego, info = embed_lsb(img, args.message.encode('utf-8'), args.key, cfg)
    stego.save(args.out)

    # Save figures
    if args.figdir:
        plot_histograms(img, stego, os.path.join(args.figdir, 'hist.png'))
        plot_lsb_planes(img, stego, os.path.join(args.figdir, 'lsb_plane.png'))

    # Save JSON
    if args.report:
        meta = {
            'input': args.input,
            'output': args.out,
            'mode': args.mode,
            'channel': args.channel,
            **info,
        }
        with open(args.report, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote stego: {args.out}")
    print(f"KL(hist) = {info['hist_kl']:.6f}")
    print(f"Used bits = {info['used_bits']}/{info['capacity_bits']}")


if __name__ == '__main__':
    main()
