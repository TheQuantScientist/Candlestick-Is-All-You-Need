# Time Series as Candlesticks

## Overview

This project explores the use of Convolutional Neural Networks (CNNs) to predict short-term price movement direction in cryptocurrencies by treating sequences of 1-minute candlestick charts as images.

The core idea is to convert historical OHLC (Open, High, Low, Close) data from Binance into small candlestick chart images using `mplfinance`, then train a simple CNN to classify whether the price will end **higher** or **lower** than the starting point of the observed window (or, in one variant, whether the last candle is bullish or bearish).

Three distinct labeling strategies are implemented in separate scripts:

1. **`full_image.py`** – "Full-window" labeling  
   Label = UP if the **last close** > **first open** of the entire window (measures overall price change over the window).

2. **`lastcandle_image.py`** – "Last-candle" labeling  
   Label = UP if the **last candle** is green (close > open). A simpler, more traditional single-candle direction prediction.

3. **`irregular_image.py`** – Irregular/sparse data experiment  
   Simulates missing minute data (60%, 80%, 95% randomly removed) while keeping fixed time windows, testing robustness to incomplete time series.

All experiments use the same CNN architecture, training regime, and evaluation metrics across multiple coins, window sizes, and time periods.

## Project Structure

```
crypto_research_minute/                  # Baseline (last-candle & full-window)
crypto_research_minute_fullimage/        # Full-window labeling results
crypto_research_minute_irregular/        # Irregular missing data results
full_image.py
lastcandle_image.py
irregular_image.py
README.md
```

Each output directory contains per-coin subfolders with:
- `raw_data/` – Downloaded 1-minute OHLC CSVs
- `images/`   – Generated 64×64 px candlestick images (DPI=32)
- `models/`   – Saved Keras `.h5` model files
- `results/`  – Text files with accuracy, F1, recall, AUROC, AUPRC

## Key Design Choices

- **Coins**: BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT
- **Time periods**: 7, 14, 21, 28 days (1–4 weeks)
- **Image window sizes**: 5, 15, 30 consecutive minutes rendered in each image
- **Image size**: 64×64 pixels (resized after generation)
- **CNN**: Lightweight 3-conv-layer model with dropout and class weighting
- **Training**: 10 epochs, binary cross-entropy, balanced class weights
- **Labeling variations**: As described above
- **Experiments**:
  - **Experiment I**: Train and test on matching period lengths (7→7, 14→14, etc.)
  - **Experiment II**: Train on 1-week data, test on 2-, 3-, 4-week periods (generalization test)

## Requirements

```bash
pip install requests pandas mplfinance matplotlib numpy pillow tensorflow scikit-learn
```

Tested with:
- Python 3.9+
- TensorFlow 2.x
- matplotlib 3.5+

## Usage

Each script can be run independently:

```bash
python lastcandle_image.py      # Baseline: predict last candle direction
python full_image.py            # Full-window price change prediction
python irregular_image.py       # Sparse/irregular data experiments
```
The scripts include checks to avoid re-downloading data, regenerating images, or retraining models if files already exist. They will resume or skip completed parts automatically.
