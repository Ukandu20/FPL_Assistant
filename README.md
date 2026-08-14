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

Processing version `1.6.0` publishes the event feed's direct chance-quality and
context signals throughout the player-match, player-season, team-match, and
team-season table families. These include big chances created and taken, big
chance conversion, error severity, completed crosses and pass-subtype success,
assisted/first-touch/one-on-one/transition/set-piece shot splits, last-man and
spatial defensive actions, high possessions won, defensive-third possessions
lost, box entries by pass or inferred carry, carries, progressive carries,
team big chances conceded, corners won, and offsides provoked. Season success
rates are recomputed from summed numerators and denominators rather than
averaging match percentages.

The same contract also publishes penalty wins/concessions, completed corners
and throw-ins with recomputed success rates, average team age,
player height/weight, pass distance and direction, intentional/shot assists,
shot body-part/technique/placement splits, goalkeeper save-location and
distribution splits, shielding/overrun/good-skill actions, native dribbles
lost, disallowed goals, and formation-change counts. Team schedules preserve
the provider manager and country metadata when supplied.

The raw field named `possession` is a minute-indexed activity map flattened by
the scraper, not a team possession percentage, so version 1.6 deliberately
does not publish it as possession share.

Spatial definitions use WhoScored's team-relative 0-100 pitch coordinates.
The attacking third begins at `x >= 66.7`; the defensive third is `x < 33.3`;
and the own penalty area is `x <= 17` with `y` between 21 and 79. A high
possession win is a recovery, interception, or successful tackle in the
attacking third. A defensive-third loss is an unsuccessful pass or take-on,
an error, or a dispossession there. A box entry by pass must start outside and
finish inside the opponent penalty area. `big_chances_created` and
`key_passes` honor the provider qualifier on any creating action, including a
touch or rebound shot, rather than incorrectly restricting creation to passes.
`shots_from_set_piece` is the complement of `open_play_shots` and includes
corner, free-kick, and penalty contexts; the narrower context columns remain
available separately.

`shot_creating_actions` and `goal_creating_actions` count the direct player
identified by WhoScored's assisted-shot `related_player_id`. They do not infer
a second preceding action or a possession chain. Carries are conservatively
inferred between consecutive same-team on-ball events separated by 1-10
seconds and 3-60 metres; progressive carries advance at least 10 WhoScored
x-coordinate units. A carry box entry starts outside and ends inside the
opponent penalty area. Advanced modelling such as xT, passing networks, and
broader possession-chain attribution is not part of this cleaning contract.

Processing version `1.7.0` adds team defensive exposure by reversing the
opponent's match row. It publishes total, on-target, off-target, blocked,
post, box, outside-box, headed, open-play, set-piece, corner, direct-free-kick,
penalty, and big-chance shots against. `shots_conceded` is an alias of
`shots_against`. Box-entry exposure is split into pass and inferred-carry
entries; `box_entries_against`, `box_entries_allowed`, and
`box_entries_conceded` are definitionally identical aliases, with corresponding
`*_by_pass_allowed` and `*_by_carry_allowed` aliases.

The cleaner also writes
`data/processed/whoscored/<league>/<season>/player_season/roles.csv`. This is an
observed, season-specific set-piece hierarchy derived only from that season's
events. Supported roles are `corner`, `penalty`, `direct_free_kick`,
`free_kick`, `indirect_free_kick`, and `long_throw`. Corners always retain the
single `corner` role; `side` identifies `left` or `right`. Rankings use recent
events, attempts, and team opportunities in matches where the player recorded
minutes. The artifact includes primary/secondary/backup ranks, confidence and
provenance. It is never seeded from a prior season, so a preseason with no
events publishes a schema-valid, zero-row `roles.csv`.

### Bootstrap a new season and publish ClubElo fixtures

At preseason, build the canonical fixture calendar directly from the official
FPL schedule. This removes the circular dependency between the fixture calendar
and cleaned WhoScored data:

```powershell
python -m fpl_assistant.providers.fbref.integrate.fixtures_meta_builder `
  --bootstrap `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fpl-root "data/raw/fpl/ENG-Premier League" `
  --force
```

The bootstrap calendar has two team-perspective rows per match and stable
`match_id` values based on league, season, home team, and away team. Kickoff
changes therefore do not change match identity. With no matches played yet,
publish WhoScored's schedule as an explicitly partial dataset:

```powershell
python -m fpl_assistant.providers.whoscored.clean.whoscored_cleaner `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --allow-partial `
  --force
```

Then publish the ClubElo-owned fixture schedule:

```powershell
python -m fpl_assistant.providers.clubelo.clean.clubelo_understat_enricher `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fixture-root "data/processed/registry/fixtures"
```

This writes `data/processed/clubelo/<league>/<season>/schedule.csv` with dynamic
`elo_pre_match` values and season-frozen `elo_preseason` values for both the
team and opponent, their differences, `elo_preseason_as_of`, and
`elo_provider`. The preseason cutoff is the day before the season's earliest
scheduled fixture. Coverage and missing-rating audits are written beside the
schedule under `audits/`.
