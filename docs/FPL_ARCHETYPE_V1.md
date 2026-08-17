# FPL player archetypes V1

This package implements the catalogue in
`docs/fpl_archetype_scheme_catalogue.md`. The catalogue is authoritative; the
decision workbook is supporting context. Archetypes describe historical player
behaviour and evidence. They are not the final FPL points forecast.

## Architecture

The implementation lives in `src/fpl_assistant/archetypes/`:

- `adapters.py` enforces provider-owned field definitions and records field
  provenance. It will not accept, for example, FPL as the source of Understat
  `npxg`.
- `schema.py` defines canonical player-match and output contracts. Existing
  canonical identity, match, roster, fixture, and DNP modules remain the source
  of stable player, club, match, gameweek, and availability keys.
- `preprocessing.py`, `temporal.py`, and `scoring.py` implement position-relative
  rates, 1st/99th winsorization, transformations, past-only standardization,
  25/30/45 evidence windows, separate small-sample shrinkage, and base scores.
- `usage.py` implements the latest-six-team-match usage model with a three-match
  half-life and mutually exclusive state boundaries.
- `families.py` implements fixture, venue, return-shape, value, and risk states.
- `clean_sheets.py` implements Clean-Sheet Specialist.
- `composites.py` resolves exactly one position-specific production composite
  after base-component activation.
- `transitions.py` applies hysteresis, transfer, position-change, injury, and
  absence rules.
- `team_ratings.py` records immutable pre-match overall Elo, attack, and defence
  ratings and applies sequential normalized xG/goals MAP updates.
- `pipeline.py`, `persistence.py`, and `cli.py` build and preserve versioned
  snapshots.
- `validation.py` and `validation_cli.py` implement chronological evaluation,
  baseline comparison, controlled parameter selection, and release gates.

All formulas, fixed weights, thresholds, grids, provider ownership, confidence
bands, and deferred component IDs are centralized in
`config/fpl_archetypes_v1.json`. A validated-but-unreleased parameter candidate
is recorded under `config/archetypes/1.0.0/selected_parameters.json`.

## Canonical input

The snapshot command accepts CSV, Parquet, or JSONL. `player_matches` must have
one row per player and completed fixture, stable `player_id` and `match_id`, a
UTC `kickoff_utc`, season, official FPL position at the snapshot date, and
minutes. Provider fields must retain their named definitions:

- FPL: minutes, starts, availability, unmultiplied points, price, cards, saves,
  and official outcomes.
- Understat: xG, npxG, and xA.
- WhoScored: shots, chance creation, defensive actions, shots faced, and detailed
  availability when present.

Missing core fields produce `no_score` (`score_0_100 = null`) with a reason in
`missing_data_flags`. Secondary metrics are reweighted only when at least 70%
of planned weight remains. Missing values are never silently changed to zero.

The optional team-match table has one row per match with home/away canonical
team IDs, goals, xG, season, gameweek, and kickoff. The Understat two-row format
can be converted with `understat_team_rows_to_matches`.

## Build a snapshot

For the processed Premier League provider data in this repository, use the
end-to-end publisher. It discovers seasons with matching FPL, Understat, and
WhoScored coverage, builds the canonical join, validates it, selects the prior
snapshot for hysteresis, and writes directly to the directory consumed by the
Streamlit app:

```powershell
python -m fpl_assistant.archetypes.publish_cli `
  --current-season 2026-2027 `
  --as-of 2026-08-17T08:15:00Z `
  --strict-input-contract
```

The installed console command is equivalent:

```powershell
fpl-archetype-publish --current-season 2026-2027 --as-of <UTC-timestamp>
```

The input builder uses FPL's full player/team-match grid for minutes, starts,
official outcomes and price; derives Understat npxG and non-penalty goals from
shot events; and joins WhoScored shooting, chance-creation, defensive and
goalkeeper evidence. Provider coverage, field provenance, duplicate resolution
and excluded identity collisions are stored as `field_provenance.jsonl` and
`input_build_audit.jsonl` in the immutable snapshot.

Use the lower-level command below only when supplying a canonical join from
another process.

```powershell
python -m fpl_assistant.archetypes.cli `
  --player-matches data/processed/canonical/player_match.csv `
  --team-matches data/processed/canonical/team_match.csv `
  --player-values data/processed/canonical/player_value.csv `
  --previous-snapshot data/processed/archetypes/previous/archetypes.jsonl `
  --current-season 2025-2026 `
  --as-of 2026-05-25T14:00:00Z
```

Outputs are partitioned by semantic model version and snapshot timestamp:

```text
data/processed/archetypes/
  model_version=1.0.0/
    snapshot=2026-05-25T14-00-00Z/
      archetypes.jsonl
      team_ratings.jsonl
      player_match_evidence.jsonl
      team_match_evidence.jsonl
      player_value_evidence.jsonl
      production_component_evidence.jsonl
      family_calculation_evidence.jsonl
      field_provenance.jsonl
      input_build_audit.jsonl
      manifest.json
```

The snapshot retains both results and their complete lead-up data:

- `player_match_evidence.jsonl` is the canonical, pre-snapshot match history,
  including window assignment, eligibility, exclusions, provider fields, and
  merged pre-match team context.
- `production_component_evidence.jsonl` is a calculation ledger containing
  raw window rates, transformations, winsorized values, positional z-scores,
  metric weights, temporal weights, raw and shrunk scores, confidence inputs,
  and post-transition decisions.
- `family_calculation_evidence.jsonl` retains usage probabilities, fixture and
  venue effects, return-shape statistics, clean-sheet inputs, value
  percentiles, and risk calculations.
- `team_match_evidence.jsonl` and `player_value_evidence.jsonl` preserve the
  other supplied inputs. `team_ratings.jsonl` remains the immutable pre-match
  team-rating sequence.
- `manifest.json` records every artifact's SHA-256 hash, row count, and column
  list.

An existing snapshot can be written again only when its bytes are identical.
A conflicting write fails instead of erasing historical output. Supply the
previous snapshot to apply two-update entry/exit and trend persistence.

The Streamlit player dashboard automatically selects the newest completed V1
snapshot within the chosen season. Its **Profile** tab shows the production
composite, usage state, every behaviour/value/risk result, and an expandable
view of the persisted calculation and player-match evidence. If no V1 snapshot
exists for that season, the app continues to show the legacy season profile.

## Validate

```powershell
python -m fpl_assistant.archetypes.validation_cli
```

The command uses processed Understat team-match seasons, selects parameters on
past seasons, and scores the latest season chronologically. It compares xG and
goal predictions with league-average, rolling xG/xGA, rolling goals/conceded,
and Elo-only baselines. Release requires at least 2% improvement, wins in 70%
of folds, no material calibration deterioration, and at least two training
seasons.

The current repository data produced a failed release result, preserved under
`artifacts/archetypes/1.0.0/validation/team_ratings/`. The held-out 2025-2026
xG result was 2.75% worse than the strongest baseline with wins in 26.3% of
folds. Goals improved 1.71% with wins in 63.2% of folds. Only one complete
season remained for training after the holdout, below the two-season minimum.
These failures are blockers for production release; V1 rules were not changed
to make the report pass.

The generic validation API also reports results by position, season,
confidence band, and evidence level for player-archetype prediction tables.
Complete player-family validation still requires canonical multi-season
WhoScored, Understat, FPL, price, and availability joins with future targets.
The per-archetype target and blocker audit is versioned in
`config/archetypes/1.0.0/validation_status.json`.

## Deferred scope

Sweeper Keeper, Distributor, every goalkeeper composite requiring either
component, Big-Game Performer, and the other catalogue-deferred concepts are
not calculated or displayed. The only V1 goalkeeper production composite is
Shot Stopper.

## Extend safely

Add a new provider field through an explicit adapter and ownership rule, update
the canonical contract, add its validated configuration, and test missing and
invalid values. Model-definition changes require a new semantic version and a
new output partition. Do not backfill an old model-version partition except for
data corrections, and never fit preprocessing or parameters with observations
after the requested snapshot cutoff.
