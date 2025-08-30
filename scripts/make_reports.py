from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Tuple

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.lsb import embed_lsb, extract_lsb, LSBConfig
from src.viz import plot_histograms, plot_lsb_planes
from src.analysis import chi_square_lsb, sample_pair_estimate
from src.perturb import jpeg_compress, resize as resize_img, rotate as rotate_img, blur as blur_img


def psnr(img1: Image.Image, img2: Image.Image) -> float:
    a = np.array(img1, dtype=np.float64)
    b = np.array(img2, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 99.0
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def read_text_bytes(path: Path) -> bytes:
    return path.read_text(encoding='utf-8', errors='ignore').encode('utf-8')


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def gen_q1(report_dir: Path, cover_path: Path, key: str) -> Tuple[Path, dict]:
    out_dir = report_dir / 'artifacts'
    ensure_dir(out_dir)

    img = Image.open(cover_path).convert('RGB')
    message = '版权水印演示：LSB + 样本对分析 + ±1匹配 (Q1)'
    cfg = LSBConfig(mode='match')
    stego, info = embed_lsb(img, message.encode('utf-8'), key, cfg)
    sp_path = out_dir / 'SP.png'
    stego.save(sp_path)

    # Figures
    plot_histograms(img, stego, str(out_dir / 'hist.png'))
    plot_lsb_planes(img, stego, str(out_dir / 'lsb_plane.png'))

    # Metrics
    met = {
        'psnr': float(psnr(img, stego)),
        'kl_hist': float(info['hist_kl']),
        'used_bits': int(info['used_bits']),
        'capacity_bits': int(info['capacity_bits']),
        'chi_before': float(chi_square_lsb(img)),
        'chi_after': float(chi_square_lsb(stego)),
        'spa_after': float(sample_pair_estimate(stego)),
    }

    # Report
    md = report_dir / 'Q1.md'
    md.write_text((
        f"# 问题1 报告：生成与提取模型与算法\n\n"
        f"- 算法：LSB 置换与 LSB 匹配（±1），像素按密钥置乱，32bit 长度头。\n"
        f"- 载体：data/B题-附件1.jpg → 生成图 SP.png。\n"
        f"- 视觉近似：PSNR={met['psnr']:.2f} dB。\n"
        f"- 统计：KL(hist)={met['kl_hist']:.6g}；χ²(前/后)={met['chi_before']:.3g}/{met['chi_after']:.3g}；SPA(after)={met['spa_after']:.3g}。\n"
        f"- 容量：{met['used_bits']}/{met['capacity_bits']} bits 已用。\n\n"
        f"## 生成结果\n"
        f"- Stego 图：reports/artifacts/SP.png\n"
        f"- 直方图对比：reports/artifacts/hist.png\n"
        f"- LSB 平面对比：reports/artifacts/lsb_plane.png\n\n"
        f"## 源代码位置（附录）\n"
        f"- 嵌入：src/lsb.py, src/encryptor.py\n"
        f"- 提取：src/lsb.py, src/parser.py\n\n"
        f"## 提取验证\n"
        f"使用 `python -m src.parser --input reports/artifacts/SP.png --key \"{key}\" --mode match` 可恢复消息。\n"
    ), encoding='utf-8')

    return sp_path, {'metrics': met}


def gen_q2(report_dir: Path, cover_path: Path, key: str) -> dict:
    img = Image.open(cover_path).convert('RGB')
    w, h = img.size
    channels = 3  # ALL
    capacity_bits = w * h * channels
    capacity_bytes = (capacity_bits - 32) // 8

    law_path = ROOT / 'airef' / '法案.md'
    law_bytes = read_text_bytes(law_path)
    law_len = len(law_bytes)

    feasible = law_len <= capacity_bytes
    embed_bytes = min(law_len, capacity_bytes)

    md = report_dir / 'Q2.md'
    md.write_text((
        f"# 问题2 报告：整部法律文本嵌入可行性\n\n"
        f"- 图片尺寸：{w}×{h}，通道：3，理论容量≈{capacity_bits} bits（去 32bit 头部后 ≈ {capacity_bytes} bytes）。\n"
        f"- 法律文本路径：airef/法案.md，大小：{law_len} bytes。\n"
        f"- 结论：{('可以全部嵌入' if feasible else '无法全部嵌入')}；最多可嵌入 {embed_bytes} bytes（约 {embed_bytes//3} 个汉字，按 UTF-8 粗估）。\n\n"
        f"## 建议\n"
        f"- 如需全部嵌入：\n"
        f"  - 降低冗余（压缩/摘要分块）；\n"
        f"  - 多图分片嵌入；\n"
        f"  - 改用 JPEG 域隐写提高鲁棒性并结合纠删编码。\n"
    ), encoding='utf-8')

    return {
        'capacity_bits': int(capacity_bits),
        'capacity_bytes': int(capacity_bytes),
        'law_bytes': int(law_len),
        'feasible': bool(feasible),
    }


def gen_q3(report_dir: Path, sp_path: Path, key: str) -> dict:
    img0 = Image.open(sp_path).convert('RGB')
    msg = '版权水印演示：LSB + 样本对分析 + ±1匹配 (Q1)'
    cfg = LSBConfig(mode='match')

    trials = []
    def attempt(tag: str, im: Image.Image):
        try:
            rec = extract_lsb(im, key, cfg).decode('utf-8', errors='ignore')
            ok = (rec == msg)
            trials.append((tag, ok, len(rec)))
        except Exception:
            trials.append((tag, False, 0))

    # Apply perturbations
    attempt('orig', img0)
    for q in [95, 85, 75]:
        attempt(f'jpeg{q}', jpeg_compress(img0, q))
    attempt('resize0.75', resize_img(img0, 0.75))
    attempt('rotate2', rotate_img(img0, 2.0))
    attempt('blur1', blur_img(img0, 1.0))

    lines = ['| 场景 | 提取成功 | 恢复长度 |', '|---|---|---|']
    for tag, ok, ln in trials:
        lines.append(f"| {tag} | {'是' if ok else '否'} | {ln} |")

    md = report_dir / 'Q3.md'
    md.write_text((
        "# 问题3 报告：压缩/格式/几何变换鲁棒性\n\n"
        "- 结论：空域 LSB/LSBM 对有损压缩与几何变换基本不鲁棒，原图可恢复，JPEG 与几何/模糊后大多失败。\n\n"
        "## 实验结果\n"
        + '\n'.join(lines) + "\n\n"
        "## 改进思路\n"
        "- 改为 JPEG 频域隐写（J-UNIWARD/UED/UERD 等）。\n"
        "- 引入纠删编码、重复扩频与同步标记，缓解位翻转与丢失。\n"
        "- 自适应嵌入：边缘/纹理区优先，降低可检性与破坏风险。\n"
    ), encoding='utf-8')

    return {'trials': trials}


def gen_q4(report_dir: Path) -> None:
    md = report_dir / 'Q4.md'
    md.write_text(
        "# 问题4 报告：推广到其他图片的注意事项（至多三条）\n\n"
        "1) 嵌入率控制与区域自适应：优先在纹理/边缘区域嵌入，降低整体嵌入率，采用 ±1 匹配减少统计痕迹。\n"
        "2) 流程与格式：避免在有损链路（如社交平台二次 JPEG）中传输 PNG 隐写结果；若不可避免，改用 JPEG 域隐写。\n"
        "3) 密钥与取证：密钥派生与像素置乱避免复用；保留哈希、时间戳与报告以便版权争议取证。\n",
        encoding='utf-8'
    )


def main():
    report_dir = ROOT / 'reports'
    ensure_dir(report_dir)

    cover_path = ROOT / 'data' / 'B题-附件1.jpg'
    key = 'sz2023b-key'

    sp_path, q1info = gen_q1(report_dir, cover_path, key)
    q2info = gen_q2(report_dir, cover_path, key)
    q3info = gen_q3(report_dir, sp_path, key)
    gen_q4(report_dir)

    # Index
    (report_dir / 'index.md').write_text(
        "# 报告索引\n"
        "- Q1：reports/Q1.md\n"
        "- Q2：reports/Q2.md\n"
        "- Q3：reports/Q3.md\n"
        "- Q4：reports/Q4.md\n",
        encoding='utf-8'
    )

    # JSON summary
    (report_dir / 'summary.json').write_text(json.dumps({
        'Q1': q1info,
        'Q2': q2info,
        'Q3': q3info,
    }, ensure_ascii=False, indent=2), encoding='utf-8')

    print('Reports written to:', report_dir)


if __name__ == '__main__':
    main()
