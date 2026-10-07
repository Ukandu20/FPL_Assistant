# FPL Expected-Minutes V2.0 Operations

> Governing specification: [hardened V2 contract](FPL_expected_minutes_V2_implementation_contract_HARDENED.md).
> Prerequisite: [eligibility and DNP publication](FPL_ELIGIBILITY_BACKFILL.md).
> Dates, GW01, RUN_ID, and example paths below are illustrative. Supply the
> intended forecast cutoff and artifacts; do not reuse an old cutoff for a live run.

The implementation lives in `fpl_assistant.minutes_v2`, separate from V1 so
the benchmark and rollback path remain reproducible. Canonical configuration is
`config/minutes_v2.json`; rolling seasons requires data/config updates, not model
source changes.

## Architecture and data boundary

V2.0 trains separate GK and pooled-outfield families for start probability,
starter duration, cameo probability for eligible non-starters, cameo duration,
and direct P60. Position is categorical. Missing evidence remains missing and is
described by `history_matches`, `season_history_matches`, and `cold_start`.

Expected minutes use only the pure mixture:

```text
p_start * minutes_if_start
+ (1 - p_start) * p_cameo * minutes_if_cameo
```

Only physical clipping is applied, with observable `fallback_flags`. Confirmed
absences use a separate timestamped override retaining raw/final predictions,
reason, source, and timestamp. Post-cutoff override knowledge is rejected.

Canonical labels cover 2022–2023 through 2025–2026. `fallback` and `imputed`
starter labels are excluded. 2020–2021 and 2021–2022 remain diagnostic-only
until an automated audit or verified backfill proves their labels reliable.
Historical `eligibility_timestamp_safe=False` is retained as a retrospective
proxy; forward inference requires timestamp-safe eligibility known by cutoff.

Canonical features exclude `played_last`, `long_gap14`, FDR, and `team_rot3`.
They exist only as isolated backtest ablations. V2.1+ priors, availability
features, congestion, manager effects, player rotation, and distributional
uncertainty are disabled.

## Commands

Audit data:

```powershell
fpl-minutes-v2 audit --prediction-cutoff 2026-08-21T17:30:00Z `
  --output data/models/minutes/v2.0/data_audit.json
```

Run deterministic walk-forward evaluation (six-GW calibration, six-GW
evaluation, six-GW steps, latest six GWs untouched):

```powershell
fpl-minutes-v2 backtest --prediction-cutoff 2026-08-21T17:30:00Z `
  --v1-predictions data/models/minutes/versions/v5/expected_minutes.csv `
  --output data/models/minutes/v2.0/backtest_predictions.csv `
  --report data/models/minutes/v2.0/backtest_report.json
```

Use `--ablation team_rot3` (or another declared ablation) for an isolated
experiment on identical folds. Reports include prescribed baselines,
MAE/RMSE/bias, full probability metrics and reliability bins, state log loss,
starter-duration/subgroup diagnostics, block-bootstrap evidence, and explicit
deferred gates when V1 state/P60 or downstream xPoints evidence is unavailable.

Install modeling dependencies and train:

```powershell
python -m pip install -e ".[modeling]"
fpl-minutes-v2 train --prediction-cutoff 2026-08-21T17:30:00Z `
  --v1-predictions data/models/minutes/versions/v5/expected_minutes.csv
```

The semantic `minutes/v2.0` run contains models, calibrators, backtest output,
and a model card with code/config/feature versions, seasons, exclusions, folds,
features, hyperparameters, seed, counts, rates, calibration, source hashes,
label distributions, eligibility safety, metrics, and timestamps.

Forecast:

```powershell
fpl-minutes-v2 forecast --artifact-dir data/models/minutes/v2.0/RUN_ID `
  --season 2026-2027 --prediction-cutoff 2026-08-21T17:30:00Z `
  --overrides data/availability/confirmed_absences.csv `
  --legacy-compatibility `
  --output data/predictions/minutes/v2/2026-2027/GW01.csv
```

`pred_exp_minutes` is canonical. Compatibility aliases are added explicitly.

## Shadow and acceptance

Generate V1 and V2 before each deadline with the same cutoff and snapshot. Each
CSV needs a `.meta.json` sidecar containing `prediction_cutoff` and identical
`input_data_identifiers`; shadow assembly rejects mismatches and roster drift.

The legacy forecaster expects numeric team IDs, `date_played`, and the old
`player_minutes_calendar.csv` name. Adapt inputs explicitly without changing V1:

```powershell
fpl-minutes-v2 prepare-v1-inputs --player-calendar path/to/player_fixture_calendar.csv `
  --fixture-calendar path/to/fixture_calendar.csv --prediction-cutoff CUTOFF `
  --gws 1 --fixtures-output path/to/v1_fixtures.csv --squads-output path/to/v1_squads.csv `
  --registry-root data/processed/registry/fixtures --history-root path/to/compat_registry `
  --history-seasons 2022-2023,2023-2024,2024-2025,2025-2026,2026-2027
```

After unchanged V1 inference, run `prepare-v1-shadow` to add the shared roster,
cutoff, snapshot identifiers, and an operational-validity check. All-zero legacy
duration output is retained for audit but marked invalid and cannot satisfy live
acceptance or rollback-readiness gates.

```powershell
fpl-minutes-v2 shadow --v1 path/to/v1.csv --v2 path/to/v2.csv `
  --prediction-cutoff 2026-08-21T17:30:00Z `
  --output data/predictions/minutes/shadow/2026-2027/GW01.csv
```

`evaluate-shadow` applies +0.002 P60 Brier, +0.005 state log-loss, and +0.01
xPoints MAE margins to the upper bound of paired 95% block-bootstrap intervals.

Engineering readiness, offline acceptance, shadow readiness, and production
approval are distinct. Production approval remains deferred until sufficient
completed 2026–2027 GWs pass every live gate, operational reliability is shown,
and rollback/cutover evidence is recorded.
