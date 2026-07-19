# Wikipedia Web Traffic — K-means Traffic Segmentation & Adaptive-MA Forecasting

> My contribution to a team MSc project on multilingual web-traffic time-series forecasting (University of Sheffield). This repo contains **only the module I built** — page-traffic segmentation with K-means, paired with a cluster-adaptive moving-average forecaster. The rest of the study (data preprocessing, PCA, anomaly detection, and the forecasting models: ARIMA/SARIMA, LSTM/GRU, Random Forest/XGBoost) was done by other team members and is not included here.

---

## About this project

The full team project forecasts daily page views across a multilingual collection of Wikipedia articles (145,063 articles over 804 days, July 2015 – September 2017). My focus was the **unsupervised segmentation** step: grouping pages with similar traffic behaviour so that a moving-average forecaster could adapt its window to each group's volatility.

**Dataset:** [Kaggle — Web Traffic Time Series Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting). The data is not committed to this repo; see [Data](#data) for where to place it.

---

## What it does

Groups pages by traffic dynamics with K-means, then forecasts each series with a moving average whose window is set per cluster.

- **Standardisation:** each series scaled to zero mean / unit variance (σ = 0 replaced with 1)
- **Features per series:** mean, standard deviation, max, min, 30-day average, and spike count (spike = value above `μ + 2σ`)
- **Clustering:** K-means with `k = 12`, `random_state = 42`, fitted on a 10,000-series sample and scaled with `StandardScaler`; the remaining series are labelled by 1-NN against the sampled fit
- **Adaptive window:** `window = clamp(0.1 × cluster_std, 3, 30)` — intended to widen for volatile clusters and tighten for stable ones
- **Forecast:** stepwise growing-window moving average over the test horizon, per cluster
- **Evaluation:** 90/10 train–test split (81-day forecast horizon), scored with MSE / RMSE / MAE, plus inverse-scaling back to original traffic counts for interpretable plots

**Results (standardised space):** overall MSE 0.5329, MAE 0.3265, RMSE 0.7300; clustering inertia 624.70 (≈0.06 average squared distance per sample, i.e. tight clusters).

> **Honest note:** in standardised space the K-means + MA approach (RMSE ≈ 0.73) *underperformed* the plain fixed-window MA baseline (≈ 0.45) in the wider team comparison. It's kept as a documented negative result — the segmentation itself is clean and tight; pairing it with a simple MA just wasn't the right forecasting fit for this data.

> **Known limitation (window saturation):** the adaptive window is driven by the *raw* traffic std. Because raw stds run into the thousands, `0.1 × std` exceeds 30 for every cluster, so the window pins to 30 across the board (visible in the cluster log — every cluster reports `window = 30`). In practice the MA is therefore fixed, not adaptive. A relative volatility measure (e.g. coefficient of variation, or ranking clusters and mapping onto the 3–30 range) would make it genuinely adapt — noted as future work.

---

## Repo structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── Data/                      # dataset goes here (not committed)
│   └── preprocessed_train_2.csv
├── src/
│   └── kmeans_adaptive_ma.py
└── outputs/                   # generated charts
```

*(Suggested layout — rename to match your actual filenames.)*

---

## Data

This repo does **not** include the dataset. The module loads a **preprocessed** CSV (`Data/preprocessed_train_2.csv`) — the cleaned file produced upstream by the team, not the raw Kaggle download. To run it:

1. Obtain the source data from the [Kaggle competition](https://www.kaggle.com/c/web-traffic-time-series-forecasting) (`train_1.csv`).
2. Apply the shared preprocessing (missing-value handling, language cohorting) to produce `preprocessed_train_2.csv`, or drop in the team's already-cleaned file.
3. Place it at `Data/preprocessed_train_2.csv`. The module handles per-series standardisation itself; the CSV's first column is `Page`, the rest are daily counts.

---

## Getting started

```bash
# clone
git clone https://github.com/AshokPitta/<repo-name>.git
cd <repo-name>

# environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run
python src/kmeans_adaptive_ma.py
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

---

## Scope & credits

This is one contributor's portion of a group MSc project. The **K-means segmentation and adaptive-MA forecasting module in this repo is my own work.** Data preprocessing, PCA, anomaly detection, and the forecasting models were developed by teammates and are intentionally excluded from this repository.

---

## Author

**Ashok Pitta** — Data Engineer / AI & ML
[GitHub](https://github.com/AshokPitta)
