深圳杯 2023B 版权保护题目

快速开始（LSB + SPA + ±1 匹配）
- 依赖：见 `requirements.txt`（需要 numpy、Pillow、matplotlib）。
- 端到端演示：运行 `scripts/demo_lsb.py` 会在 `out_lsb/` 生成隐写图片与图表。
- 命令行：
	- 嵌入：`python -m src.encryptor --input data/B题-附件1.jpg --out out_lsb/stego.png --key my-key --message "hello" --mode match --figdir out_lsb --report out_lsb/report.json`
	- 提取：`python -m src.parser --input out_lsb/stego.png --key my-key`

详细说明见 `docs/思路说明.md`。
