# Candlestick Is All You Need

<!-- ![Your Banner](results/graphs/banner.png/) -->

![High-level Design Image](results/graphs/design.png/)

The work investigates whether modern **pre-trained computer vision backbones** (MobileNetV3, EdgeNeXt, GhostNet, EfficientNet-B0, LeViT) and a very **lightweight custom CNN** can effectively predict the directional movement (Bullish/Bearish) of major cryptocurrencies using **only candlestick chart images** generated from 1-minute OHLC data, without any additional numerical features or time-series specific architectures.

## Key Contributions

- Systematic comparison of **5 strong pre-trained vision models** vs. a tiny custom CNN (~few hundred thousand parameters) on the **same image classification task**
- Three complementary experimental settings:
  - **Exp A** — Full-window labeling (last close > first open) + memory-efficient training
  - **Exp B** — Standard last-candle labeling (realistic trading signal)
  - **Exp C** — Irregular / missing data robustness (60%, 80%, 95% random omission)
- Multi-month **out-of-sample testing** on volatile periods across **BTC, ETH, BNB, XRP, ADA, DOGE**
- Train on short windows (1 week) and test on longer horizons (2–4 weeks) — **temporal generalization** study
- Extremely low memory footprint training pipelines (lazy loading + TF Dataset / PyTorch DataLoader)

## Repository Structure

```markdown
thequantscientist-candlestick-is-all-you-need/
├── requirements.txt               # Core dependencies (minimal & optional torch/timm)
└── src/
    ├── architecture/
    │   └── small_cnn.py           # Lightweight custom CNN (used in Exp A/B/C Small-CNN variant)
    └── imaging/
        ├── Pre-trained/           # Experiments using timm & TF pre-trained models
        │   ├── Experiment_A.py    # Full-window labeling + reuse regular images
        │   ├── Experiment_B.py    # Standard last-candle labeling
        │   └── Experiment_C.py    # Irregular / missing data (60–95%)
        └── Small-CNN/             # Same three experiments — but using custom tiny CNN
            ├── Experiment_A.py
            ├── Experiment_B.py
            └── Experiment_C.py
```

## Experimental Settings Summary

| Experiment | Labeling Rule                        | Data Completeness | Model Family          | Purpose                              |
|------------|--------------------------------------|-------------------|-----------------------|--------------------------------------|
| **A**      | Full window (last close > first open) | 100%              | Pre-trained + Small CNN | Strongest possible signal             |
| **B**      | Last candle (close > open)           | 100%              | Pre-trained + Small CNN | Realistic trading-label baseline     |
| **C**      | Last candle                          | 5–40% kept        | Pre-trained + Small CNN | Robustness to missing / irregular data |

All experiments test:

- Multiple **window sizes** (5, 15, 30 candles)
- Multiple **training periods** (1 week → test on 1–4 weeks)
- **6 major coins** with different market regimes

## Requirements

```text
# Core
pandas numpy requests tqdm Pillow matplotlib mplfinance

# ML
scikit-learn

# TensorFlow (EfficientNet + custom CNN)
tensorflow>=2.10

# Optional — for timm models (MobileNetV3, EdgeNeXt, GhostNet, LeViT)
torch torchvision timm
```

Install:

```bash
pip install -r requirements.txt
# If using GPU + timm models → install matching torch+cuda version manually
```

## Usage

Each experiment script supports the same powerful argument interface:

```bash
# Example: run full pre-trained pipeline for BTC, MobileNetV3, 5-candle windows, only 7-day periods
python src/imaging/Pre-trained/Experiment_B.py --model mobilenetv3 --coin BTCUSDT --window 5 --time-length 7

# Run only Experiment II (1-week train → longer test) with small CNN
python src/imaging/Small-CNN/Experiment_A.py --exp2-only

# Run irregular data experiment with 80% missing data
python src/imaging/Pre-trained/Experiment_C.py --model edgenext --missing 0.8

# See all options
python src/imaging/Pre-trained/Experiment_B.py --help
```

**Important:** Scripts are **idempotent** — they skip already computed raw data, images, models, and result files.

## Reproducibility Notes

- All models are either **fully pre-trained** (ImageNet) or **very small** → training is fast even on modest hardware
- Lazy loading prevents RAM explosion (tested with <1 GB peak usage per job)
- Results are deterministically saved in per-coin subfolders: `results/`, `models/`
- Raw OHLCV data is fetched from **Binance public API** (no private keys needed)

## Acknowledgments

Built with:

- [mplfinance](https://github.com/matplotlib/mplfinance) — beautiful candlestick rendering  
- [timm](https://github.com/huggingface/pytorch-image-models) — excellent pre-trained vision backbones  
- Binance Public API — 1-minute OHLCV data
