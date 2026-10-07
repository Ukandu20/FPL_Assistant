# Archetyping and modelling: methodology and execution

This is the companion to [Complete data pipeline](COMPLETE_DATA_PIPELINE.md).
It begins after processed providers, FPL rosters, canonical fixtures, expanded
eligibility calendars, and team/player form are published and assurance passes.
It explains the roles of archetypes and predictive models, then gives the
current package commands in dependency order.

Scope: season profiles, V1 archetypes, V2 expected minutes, goals/assists,
defense, saves, discipline, expected points, and the hand-off to captain and
squad decisions. These instructions do not change the governing model contracts
or certify the existing models for production. Commands were checked against
source; this document was not validated by running a complete training cycle.

## 1. How the pieces fit

| Layer | Question answered | Main output | Downstream use |
|---|---|---|---|
| Season profiles | How has this player produced relative to their position? | `analytics/player_profiles.csv` | Legacy Player Card evidence |
| V1 archetypes | What behaviour, usage, and production style does the evidence support? | Versioned `archetypes.jsonl` and calculation evidence | Player explanation and comparison |
| Minutes V2 | Will the player start, appear, reach 60 minutes, and for how long? | Fixture-level minutes and probability predictions | Exposure for other models |
| Goals/assists | What attacking rates and event probabilities are expected? | G/A means and probabilities | Attacking points |
| Defense | What clean-sheet, conceded-goal, and contribution outcomes are expected? | Defense probabilities/rates | Defensive points |
| Saves | How many saves might a goalkeeper make? | Save rates/counts | Goalkeeper points |
| Discipline | What negative events might occur? | Separate negative-component estimates | Research/evaluation; not automatically wired into future points |
| Expected points | How do the component forecasts translate into points? | Component breakdown and aggregate score | Ranking, captain and squad workflows |

Archetype publication and forecast modelling are parallel consumers of cleaned
data. Current model commands do not automatically consume archetype scores.
Publishing an archetype snapshot neither trains a minutes model nor produces
an FPL points forecast. Do not substitute a style score for an expected return.

The operational dependencies are:

```text
Processed providers + FPL + canonical calendars + form
  -> season profiles (optional legacy artifact)
  -> archetype snapshot + archetype validation
  -> minutes audit -> chronological backtest -> trained V2 -> future minutes
       -> goals/assists, defense and saves forecasts
       -> reviewed points combination -> ranking/decision inputs
```

Component model training uses historical out-of-sample minutes predictions,
not the current gameweek forecast. Do training/evaluation when creating a model
version; routine weekly publication reuses explicitly selected model artifacts.

## 2. Shared setup and cutoff

Use PowerShell from the repository root in the project Python environment:

```powershell
python -m pip install -e ".[modeling]"
$env:PYTHONUTF8 = "1"
$season = "2026-2027"
$league = "ENG-Premier League"
$gw = [int](Read-Host "Target gameweek number")
$cutoff = Read-Host "Prediction cutoff in UTC ISO format, ending in Z"
$historySeasons = "2022-2023,2023-2024,2024-2025,2025-2026"
$formVersion = "latest"
$minutesFile = "data/predictions/minutes/v2/$season/GW{0:D2}.csv" -f $gw
```

Set the actual intended cutoff, not a copied historical deadline. Every feature,
roster, availability override, and fixture snapshot used for a forecast must be
available by that cutoff. Reconstructing old predictions from today's snapshots
is not a valid live backtest. Pending fixtures have null outcomes, and the
eligibility loader excludes them from training labels.

`latest` is convenient for a routine current run. For a reproducible evaluation,
record hashes or use an immutable form version and immutable input snapshots.
Check season coverage before retaining the example historical season list.
Consult `config/minutes_v2.json` for the actual V2 label seasons and feature set.

## 3. Optional legacy season production profiles

Counts become per-90 rates and are shrunk toward a position-specific mean:

```text
reliability = minutes / (minutes + prior_minutes)
adjusted_rate = reliability * player_rate + (1 - reliability) * position_mean
```

The adjusted rates are standardized within position and combined into bounded
scores and percentiles. Reliability is not applied a second time to the final
composite. Low evidence and missing provider data have distinct states.

```powershell
python -m fpl_assistant.providers.fpl.pipelines.player_profiles `
  --league $league `
  --season $season
```

Output: `data/processed/fpl/<league>/<season>/analytics/player_profiles.csv`.
This is the legacy season artifact; it is not the versioned V1 archetype system.
See [Player profiles](PLAYER_PROFILES.md) for its thresholds and schema.

## 4. Publish V1 archetypes

The governing [catalogue](fpl_archetype_scheme_catalogue.md) defines production,
usage, fixture/venue, return-shape, value, and risk families. The implementation
uses position-relative transformations, historical evidence windows, small-sample
shrinkage, confidence, and persistent transition rules. Base components may
coexist internally; production-composite precedence selects one display label.

Usage uses a previous-season baseline before the new season starts, then recent
team-match evidence. It must not advance hysteresis simply because the same
unchanged snapshot is rebuilt. Missing core metrics suppress the affected score;
a lack of data is not zero production. Deferred goalkeeper components remain
outside V1.

```powershell
python -m fpl_assistant.archetypes.publish_cli `
  --league $league `
  --current-season $season `
  --as-of $cutoff `
  --strict-input-contract
```

The publisher discovers joinable evidence seasons, constructs the provider join,
selects an earlier snapshot for transitions, and writes immutable snapshots:

```text
data/processed/archetypes/model_version=<version>/snapshot=<timestamp>/
  archetypes.jsonl
  team_ratings.jsonl
  player_match_evidence.jsonl
  production_component_evidence.jsonl
  family_calculation_evidence.jsonl
  field_provenance.jsonl
  input_build_audit.jsonl
  manifest.json
```

CSV and Parquet companions are also published. Inspect missing-data flags,
identity exclusions, confidence, provenance, and manifest hashes. A conflicting
write to an existing immutable snapshot is an error, not permission to replace
history.

### Validate the archetype team-rating component

```powershell
python -m fpl_assistant.archetypes.validation_cli
```

This command evaluates team-rating parameter choices against chronological
baselines. It does not validate every player archetype family. Review family
validation status separately in `config/archetypes/1.0.0/validation_status.json`.
A published snapshot is not proof that all release gates passed. Detailed model
structure, prior recorded failures, and output contracts are in
[Archetypes V1](FPL_ARCHETYPE_V1.md).

## 5. Expected-minutes V2: audit, evaluate, train

V2 models separate goalkeeper and pooled-outfield families with start probability,
starter duration, cameo probability conditional on not starting, cameo duration,
and direct probability of reaching 60 minutes. Expected minutes follow:

```text
E[minutes] = P(start) * E[minutes | start]
           + (1 - P(start)) * P(cameo | not start) * E[minutes | cameo]
```

`P(cameo | not start)` is not unconditional appearance probability. Retain
explicit `p_play` and calibrated P60 when passing predictions downstream.
Confirmed absences are timestamped overrides preserving raw and final forecasts;
post-cutoff knowledge cannot be used as a pre-cutoff feature.

The canonical configuration excludes untrusted starter labels, uses deterministic
chronological folds, and keeps deferred extensions out of V2.0. Report minutes
MAE/RMSE/bias as well as probability calibration, Brier scores, state log loss,
subgroup performance, and comparisons with simple baselines.

```powershell
python -m fpl_assistant.minutes_v2.cli audit `
  --prediction-cutoff $cutoff `
  --output "data/models/minutes/v2.0/data_audit.json"

python -m fpl_assistant.minutes_v2.cli backtest `
  --prediction-cutoff $cutoff `
  --output "data/models/minutes/v2.0/backtest_predictions.csv" `
  --report "data/models/minutes/v2.0/backtest_report.json"

python -m fpl_assistant.minutes_v2.cli train `
  --prediction-cutoff $cutoff
```

Run blocks sequentially and inspect the reports before training or promoting a
candidate. Add `--v1-predictions` with a verified historical V1 benchmark file to
backtest/train when available. Without that evidence, comparative acceptance
checks may remain deferred. Do not supply the current forward `GW05.csv` as a
historical out-of-sample benchmark.

Training writes a run directory below `data/models/minutes/v2.0/` containing
models, calibrators, and a model card. Select the actual run, not an assumed
latest path. The [hardened contract](FPL_expected_minutes_V2_implementation_contract_HARDENED.md)
governs methods; [V2 operations](FPL_EXPECTED_MINUTES_V2.md) governs detailed
shadow and compatibility procedures.

## 6. Forecast minutes for the chosen gameweek

```powershell
$minutesModelDir = Read-Host "Path to the selected trained minutes V2 run directory"
python -m fpl_assistant.minutes_v2.cli forecast `
  --artifact-dir $minutesModelDir `
  --season $season `
  --prediction-cutoff $cutoff `
  --gws $gw `
  --legacy-compatibility `
  --output $minutesFile
```

Use `--overrides` only with a real, timestamped confirmed-absence file. Omitting
it does not create an empty or fabricated availability source. Inspect the CSV
and metadata sidecar, target fixtures, roster coverage, fallback flags, and the
input identifiers. Forecasts should be saved before the relevant deadline.

`--legacy-compatibility` adds `pred_minutes`, `expected_minutes`, `p_start`,
`p_cameo`, `p60`, `p_play`, and `exp_minutes_points`. It does not make every older
consumer schema-compatible. The canonical value remains `pred_exp_minutes`.

## 7. Train/evaluate the attacking, defensive, and saves components

These are older model builders under `fpl_assistant.models`. They consume
`players_form.csv`/`team_form.csv` and historical minutes predictions. Their
held-out slice is controlled by the last season and `--first-test-gw`; it is not
a replacement for the V2 chronological acceptance procedure.

| Component | Method in the current implementation | Inspect |
|---|---|---|
| Goals/assists | Position-specific per-90 models, optional Poisson heads, exposure scaling and event probabilities | MAE, event Brier/calibration, missing-feature audits |
| Defense | Team clean-sheet/conceded-goal models and player defensive-contribution model | Team calibration, DCP labels/coverage, P60 join coverage |
| Saves | Goalkeeper save-rate model scaled by expected exposure | Save count error, calibration/dispersion, keeper coverage |

### Historical minutes input contract

Provide genuinely out-of-sample predictions covering the component evaluation
rows. The legacy loaders use date and GW join keys; check joins against the
historical form data rather than renaming a future forecast into place.

The standard V2 compatibility helper does not add all trainer-specific aliases:

| Older consumer | Required compatibility check |
|---|---|
| Defense trainer | Expects `prob_played60_cal`, `prob_played60_raw`, or `prob_played60`; ordinary `p60` alone is not read there |
| G/A optional mixture | Looks for `pred_start_head` and `pred_bench_cameo_head` as well as start/cameo probabilities |
| Component trainers | Require historical `date_played`, season, player and GW keys with appropriate target coverage |

If adapting a V2 backtest export, preserve the original and create a separate
compatibility copy. Map calibrated `p60_cal` to `prob_played60_cal`,
`pred_minutes_if_start` to `pred_start_head`, and `pred_minutes_if_cameo` to
`pred_bench_cameo_head` only when those source columns exist and the consumer's
conditional semantics agree. Keep the normal V2 aliases too. Do not infer P60
from mean minutes simply to make the loader accept the file.

The commands below require that prepared historical file:

```powershell
$historicalMinutes = Read-Host "Path to validated out-of-sample historical minutes in the trainer-compatible schema"
$firstTestGw = 26
```

### Goals and assists

```powershell
python -m fpl_assistant.models.goals_assists_model_builder `
  --seasons $historySeasons `
  --first-test-gw $firstTestGw `
  --features-root "data/processed/registry/features" `
  --form-version $formVersion `
  --minutes-preds $historicalMinutes `
  --require-pred-minutes `
  --poisson-heads `
  --model-out "data/models/goals_assists" `
  --bump-version `
  --log-level INFO
```

### Defense

```powershell
python -m fpl_assistant.models.defense_model_builder `
  --seasons $historySeasons `
  --first-test-gw $firstTestGw `
  --features-root "data/processed/registry/features" `
  --form-version $formVersion `
  --minutes-preds $historicalMinutes `
  --require-pred-minutes `
  --skip-gk `
  --model-out "data/models/defense" `
  --bump-version `
  --log-level INFO
```

### Goalkeeper saves

```powershell
python -m fpl_assistant.models.saves_model_builder `
  --seasons $historySeasons `
  --first-test-gw $firstTestGw `
  --features-root "data/processed/registry/features" `
  --form-version $formVersion `
  --minutes-preds $historicalMinutes `
  --require-pred-minutes `
  --poisson-head `
  --model-out "data/models/saves" `
  --bump-version `
  --log-level INFO
```

Inspect actual model files, metadata, metrics, medians/preprocessing artifacts,
and prediction coverage before selecting a version. A latest pointer or version
folder alone does not establish a complete trained model. Match forecast options
to the fitted artifact schema. These examples choose options, not tuned or
approved hyperparameters.

## 8. Forecast attacking, defense, and save components

Select the trained component versions explicitly. Historical features should
include the current season's completed evidence when available; the `--as-of`
cutoff must exclude future outcomes.

```powershell
$forecastHistory = "$historySeasons,$season"
$gaModelDir = Read-Host "Path to the selected goals/assists model version"
$defenseModelDir = Read-Host "Path to the selected defense model version"
$savesModelDir = Read-Host "Path to the selected saves model version"
```

```powershell
python -m fpl_assistant.models.goals_assists_forecast `
  --history-seasons $forecastHistory `
  --future-season $season `
  --as-of $cutoff `
  --as-of-tz UTC `
  --as-of-gw $gw `
  --n-future 1 `
  --features-root "data/processed/registry/features" `
  --form-version $formVersion `
  --fix-root "data/processed/registry/fixtures" `
  --minutes-csv $minutesFile `
  --model-dir $gaModelDir `
  --teams-json "data/processed/registry/master_teams.json" `
  --league-filter $league `
  --require-on-roster `
  --out-dir "data/predictions/goals_assists" `
  --out-format csv `
  --zero-pad-filenames `
  --log-level INFO
```

```powershell
python -m fpl_assistant.models.defense_forecast `
  --history-seasons $forecastHistory `
  --future-season $season `
  --as-of $cutoff `
  --as-of-tz UTC `
  --as-of-gw $gw `
  --n-future 1 `
  --features-root "data/processed/registry/features" `
  --form-version $formVersion `
  --fix-root "data/processed/registry/fixtures" `
  --minutes-csv $minutesFile `
  --model-dir $defenseModelDir `
  --teams-json "data/processed/registry/master_teams.json" `
  --league-filter $league `
  --require-on-roster `
  --require-pred-minutes `
  --out-dir "data/predictions/defense" `
  --out-format csv `
  --zero-pad-filenames `
  --log-level INFO
```

```powershell
python -m fpl_assistant.models.saves_forecast `
  --history-seasons $forecastHistory `
  --future-season $season `
  --as-of $cutoff `
  --as-of-tz UTC `
  --as-of-gw $gw `
  --n-future 1 `
  --features-root "data/processed/registry/features" `
  --form-version $formVersion `
  --fix-root "data/processed/registry/fixtures" `
  --minutes-csv $minutesFile `
  --model-dir $savesModelDir `
  --teams-json "data/processed/registry/master_teams.json" `
  --league-filter $league `
  --require-on-roster `
  --require-pred-minutes `
  --out-dir "data/predictions/saves" `
  --out-format csv `
  --zero-pad-filenames `
  --log-level INFO
```

Check the logged output paths. With a one-GW window, these legacy forecasters
use a window-style name such as `GW05_05.csv`, whereas the V2 example writes
`GW05.csv`. Passing `--minutes-csv` explicitly avoids accidental discovery of
another minutes version. Do not assume success if rows were dropped, feature
coverage is poor, or the intended fixture population changed.

## 9. Discipline and expected-points integration

### Discipline is a separate workflow

`discipline_model_builder` estimates negative-event components and supports
future-season inference. It is not automatically consumed by `points_forecast`.
Use its help to configure a dedicated evaluation or experiment:

```powershell
python -m fpl_assistant.models.discipline_model_builder --help
```

### Review the points combination before using it for decisions

The intended decomposition combines appearance, attacking, defensive, goalkeeper,
and negative-event contributions. Some terms are nonlinear: probability of 60
minutes matters separately from mean minutes, and save/concession thresholds
cannot generally be replaced by a points multiplier on the mean count.

The current repository has two different points combiners:

| Command | Role | Current implementation limitation |
|---|---|---|
| `expected_points_aggregator` | Versioned combination of component evaluation outputs; optional actuals | Uses a season/GW/player/team key and legacy probability fallback conventions |
| `points_forecast` | Future GW-window publication | Calculates concession/discipline fields but excludes them from the final `xPts` sum |

Neither command should be described as a validated complete scoring engine.
The future sum contains appearance, goals, assists, clean sheets, saves, and
DCP terms; it does not include a conventional bonus-points forecast. DCP bonus
and the separate 0?3 bonus award are different concepts. Audit clean-sheet
probability semantics too: some inputs are already multiplied by P60, so a
second exposure multiplication would understate that contribution.

Both older combiners use `season, gw_orig, player_id, team_id` as their main
join key, omitting canonical match identity. Double-gameweek rows therefore
need a dedicated fixture-level join/cardinality review before aggregation.
The later GW combiner cannot recover fixture distinctions lost upstream.

The following is an inspection/research publication command, not a production
approval. Select the exact files produced for the same cutoff and target window:

```powershell
$gaFile = Read-Host "Exact goals/assists forecast CSV path"
$defenseFile = Read-Host "Exact defense forecast CSV path"
$savesFile = Read-Host "Exact saves forecast CSV path"
python -m fpl_assistant.models.points_forecast `
  --minutes $minutesFile `
  --goals-assists $gaFile `
  --defense $defenseFile `
  --saves $savesFile `
  --future-season $season `
  --as-of-gw $gw `
  --n-future 1 `
  --out-dir "data/predictions/points_review" `
  --out-format csv `
  --zero-pad-filenames `
  --no-merged `
  --log-level INFO
```

The separate review directory and `--no-merged` keep this experimental output
out of the normal accumulated points publication. Resolve missing terms,
probability conventions, and match-key issues before treating it as a complete
forecast. This document records these integration gaps; it does not fix them.

### Optional GW totals after fixture-level validation

```powershell
$validatedPointsFile = Read-Host "Path to reviewed per-fixture expected-points output"
python -m fpl_assistant.models.total_points_combiner `
  --xp-csv $validatedPointsFile `
  --minutes-csv $minutesFile `
  --season $season `
  --gws $gw `
  --out-dir "data/predictions/points_review/gw_totals" `
  --auto-version `
  --log-level INFO
```

Confirm the selected points file has the column schema expected by this
combiner; the two upstream points workflows are not interchangeable merely
because both produce CSV. Review `xp_by_gw.csv` and metadata. Captain ranking
and squad/transfer optimization are subsequent decision layers, not archetypes
or forecast calibration. Their evaluation needs aligned predictions, outcomes,
prices, and squad constraints; it is outside this publication sequence.

## 10. Acceptance, shadow comparison, and weekly operation

For a new model version, retain data audits, splits, calibration evidence,
baselines, subgroup results, and immutable artifact identifiers. Compare V1
and V2 before deadlines using identical rosters, cutoffs, and input identifiers.
Follow [V2 operations](FPL_EXPECTED_MINUTES_V2.md) for `prepare-v1-inputs`,
`prepare-v1-shadow`, `shadow`, and `evaluate-shadow`. Historical benchmarks
must not be presented as forecasts saved before a live deadline.

Release review should establish:

- All required input seasons, provider fields, and canonical identities are present.
- No pending result or post-cutoff information enters training or inference features.
- Probabilities are bounded and coherent; durations and expected minutes are physical.
- Component joins preserve the intended player-fixture population and identify every missing prediction.
- Holdout and chronological metrics meet the governing contracts, including deferred gates.
- Points terms and double-gameweek behavior are explicitly verified before ranking players.
- The application reads the intended season and version and can identify missing forecasts.

A normal weekly refresh is: run the data pipeline, choose one cutoff, publish
archetypes, select already-reviewed trained artifacts, forecast minutes and
components, inspect joins and coverage, and archive outputs/metadata. Retraining
is a separate deliberate operation. Until the points integration gaps above
are resolved, stop at reviewed component outputs rather than claiming a fully
validated end-to-end expected-points publication.

## Related contracts

- [Data pipeline](COMPLETE_DATA_PIPELINE.md): upstream execution order.
- [Archetype catalogue](fpl_archetype_scheme_catalogue.md) and [V1 operations](FPL_ARCHETYPE_V1.md): archetype definitions and release status.
- [Hardened minutes contract](FPL_expected_minutes_V2_implementation_contract_HARDENED.md) and [V2 operations](FPL_EXPECTED_MINUTES_V2.md): minutes methodology and acceptance.
- [Eligibility](FPL_ELIGIBILITY_BACKFILL.md): training population and timestamp safety.
- [Historical model evaluation](MODEL_EVALUATION_REPORT.md): dated findings, not a new certification.
