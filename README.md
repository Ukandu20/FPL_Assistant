+/🧠 Fantasy Premier League ML Assistant

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
```

## Run the analytics app

Install the application dependencies and launch the canonical multipage entry
point from the repository root:

```powershell
python -m pip install -e ".[app]"
streamlit run apps/fpl/app.py
```

The public app is read-only. Model execution and optimizer controls remain in
the separate research control panel.

## Provider processing safety

- Files under `data/raw` are immutable provider inputs. Cleaning and enrichment
  outputs belong under `data/processed`.
- Understat normalizes `YYYY-YYYY` CLI input to the provider's starting year.
  Equivalent season folders cannot both supply data, and schema-only inputs
  cannot replace populated outputs.
- FBref schedules must contain dates inside the requested season. Invalid or
  wholly schema-only seasons cannot update identity or relegation registries.
- ClubElo enrichment writes copies below `data/processed/clubelo`; it never
  modifies its ClubElo or Understat inputs.
- WhoScored defensive processing requires complete schedule coverage, canonical
  player/team/match bridges, and official FPL totals unless a diagnostic bypass
  is explicitly selected.
- The native WhoScored cleaner publishes provider-owned `player_match`,
  `player_season`, `team_match`, and `team_season` table families with registry
  IDs and retained `provider_*_id` columns. Event-derived counts take precedence
  over the scraper's display-stat counters; disagreements are written to audits.
- WhoScored match roles remain provider-observed in `provider_position_match` and
  `position_detail_match`. Season primary roles are minutes-weighted, while
  `fpl_pos` is resolved independently from the season registry, the observed
  WhoScored role, the WhoScored season primary, then the latest prior registry
  position. Substitute rows therefore retain an unobserved match role but still
  receive the determined FPL classification and a provenance-labelled imputation.
- FotMob and Transfermarkt are not canonical feature sources. Before promoting
  either one, add a staging contract, identity bridges, a coverage audit, and a
  provider-to-canonical metric map.

### Clean and integrate FBref data

Use the [FBref pipeline runbook](docs/FBREF_PIPELINE.md) for the complete
PowerShell sequence: raw coverage checks, the league/season-scoped cleaner,
quarantine of source-deprecated 2025-2026 advanced outputs, canonical fixture
and match-ID integration, feature publication, and final assurance.

### Clean native WhoScored data

The production-safe default rejects incomplete coverage and unresolved
identities:

```powershell
python -m fpl_assistant.providers.whoscored.clean.whoscored_cleaner `
  --league "ENG-Premier League" `
  --season 2025-2026
```

For an explicitly partial historical or diagnostic publication:

```powershell
python -m fpl_assistant.providers.whoscored.clean.whoscored_cleaner `
  --league "ENG-Premier League" `
  --season 2025-2026 `
  --allow-partial `
  --no-strict-identities `
  --force
```

Outputs are written below
`data/processed/whoscored/<league>/<YYYY-YYYY>/`. Persistent WhoScored player,
team, and match bridges are stored below `data/processed/registry/bridges/`.
Use `--rebuild-provider-bridges` when deliberately re-evaluating WhoScored
matches after changing identity rules; rows for other providers are preserved.
