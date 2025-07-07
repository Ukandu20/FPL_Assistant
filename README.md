# 🧠 Fantasy Premier League ML Assistant

> A machine learning-powered assistant to help optimize Fantasy Premier League (FPL) team decisions each gameweek — built for interpretability and open-source reproducibility.

## 🏆 Objective

This project aims to:

- Predict FPL player performance (expected points) using interpretable models
- Recommend optimal captain picks and transfers each week
- Provide a visual dashboard to guide weekly decisions
- Serve as a public ML portfolio project, prioritizing transparency and reproducibility

## ✅ Success Criteria

- Beat my personal FPL score from the 2024/25 season
- Generate a working dashboard before Gameweek 1 of the 2025/26 season
- Publish full code, writeup, and evaluation for public review

---

## 📅 Project Timeline

| Week | Milestone |
|------|-----------|
| 1 | Setup, data collection, exploratory analysis |
| 2 | Feature engineering (form, fixtures, xG, etc.) |
| 3 | Baseline modeling (Linear, Decision Tree) |
| 4 | Pick/captain decision logic |
| 5 | Streamlit dashboard |
| 6 | Backtesting, polish, Gameweek 1 picks |

---

## 🧠 Features Engineered

- Rolling average points (last 3 matches)
- Minutes played trend
- Opponent difficulty (ELO)
- Home/away factor
- xG + xA involvement
- Value metrics (points per £)

---

## ⚙️ Technologies

| Area | Tool |
|------|------|
| Data | `pandas`, `requests`, `fpl` |
| ML | `scikit-learn`, `xgboost`, `lightgbm`, `Optuna` |
| Viz | `matplotlib`, `seaborn`, `Altair`, `Plotly`, `Streamlit` |
| Infra | `Git`, `Python 3.11`, `Jupyter`, `Streamlit`|

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/FPL_Assistant.git
cd FPL_Assistant
