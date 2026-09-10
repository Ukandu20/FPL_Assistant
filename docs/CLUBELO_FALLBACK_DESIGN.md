# Temporary ClubElo Fallback: Hardened Logic Contract

## Status

Implemented as an isolated, dry-run-by-default provider. It is not automatically
activated and does not mutate the existing ClubElo scraper or authoritative
ClubElo data.

Implementation:

`src/fpl_assistant/providers/clubelo/fallback/elo_fallback.py`

Focused tests:

`tests/test_clubelo_fallback.py`

The fallback exists only to bridge a temporary ClubElo outage. It must remain a
separate, explicitly labelled provider and be removable when ClubElo returns.

## 1. Scope and truthfulness

The first release is Premier-League-local:

- anchor ratings come from the last available ClubElo preseason values;
- only completed Premier League matches are applied;
- scores, not xG, determine Elo changes;
- future fixtures never change a rating;
- generated values are labelled `clubelo_fallback`, never `clubelo`;
- raw ClubElo files are read-only.

This is not an exact reconstruction of ClubElo. ClubElo also processes domestic
cup, European, and other eligible matches, and uses a result histogram that is
not available while its service is offline. Those differences must be visible in
the provider and audit metadata.

## 2. Current 2026-2027 input contract

### Anchor ratings

Source:

`data/processed/clubelo/ENG-Premier League/2026-2027/schedule.csv`

Required anchor fields:

- `team`
- `team_id`
- `elo_preseason`
- `elo_preseason_as_of`

Observed baseline at design time:

- 760 team-fixture rows / 380 fixtures;
- exactly 20 unique teams and 20 unique anchors;
- no missing preseason ratings;
- anchor date `2026-08-20`.

The implementation must reduce the schedule to one anchor per `team_id` and
reject conflicting team codes, anchor values, or anchor dates.

### Completed results

Initial source:

`data/processed/understat/ENG-Premier League/2026-2027/schedule.csv`

Required fields:

- `match_id`
- `game_date` and, when available, `game_time`
- `team`, `team_id`, `opp`, `opp_id`, `venue`
- `team_goals`, `opp_goals`
- `is_result`, `has_data`

Observed baseline at design time:

- 760 team-fixture rows;
- 20 completed team-side rows;
- 10 unique completed matches.

Only rows with `is_result == True` and `has_data == True` are eligible. A match
is accepted only when its two team-side rows mirror one another exactly.

## 3. Canonical match reduction

Before calculating ratings, paired team-side rows are reduced to one match row:

```text
match_id, kickoff_utc, home_team_id, away_team_id, home_goals, away_goals
```

Hard failures:

- missing or duplicate `match_id` + `team_id`;
- anything other than exactly one home and one away row per completed match;
- mismatched teams or mirrored scores;
- missing, negative, fractional, or non-finite goals;
- home and away resolving to the same team;
- a team not present in the anchor set;
- a completed match on or before the anchor timestamp;
- conflicting score revisions for the same `match_id`.

Matches are replayed from the anchor on every run. They are sorted by
`kickoff_utc`, then `match_id`, with a stable sort. Rebuilding from source makes
the process deterministic, idempotent, and safe after corrected results.

## 4. Rating equation

For a home rating `H`, away rating `A`, and fixed home-field advantage `hfa`:

```text
expected_home = 1 / (1 + 10 ** (-(H - A + hfa) / 400))
actual_home   = 1.0 for a home win, 0.5 for a draw, 0.0 for a loss
base_delta    = k * (actual_home - expected_home)
```

Initial parameters are versioned configuration, not hidden constants:

```text
formula_version = "clubelo_fallback_v1"
k = 20.0
hfa = 36.0
elo_divisor = 400.0
```

The away change is always the exact negative of the home change:

```text
new_home = H + delta
new_away = A - delta
```

This gives a strict zero-sum invariant for every match and for the league as a
whole, subject only to floating-point tolerance.

## 5. Goal-margin policy

ClubElo weights decisive results by `sqrt(goal_margin)`, but normalizes that
weight using conditional score-margin probabilities from its result histogram:

```text
margin_factor = sqrt(margin) / E[sqrt(margin) | win_or_loss, rating_context]
delta = base_delta * margin_factor
```

The denominator is not currently present in this repository. Multiplying by
`sqrt(margin)` alone is therefore not treated as exact ClubElo logic: it raises
the expected effective K and creates avoidable drift.

The hardened implementation must support two explicit modes:

1. `base` (safe default): `delta = base_delta`; no margin approximation.
2. `calibrated_margin`: use only a checked-in, versioned normalization model
   fitted on historical Premier League matches with real ClubElo transitions.

`calibrated_margin` cannot be enabled until historical replay passes the gates
in section 9. Draws always use a margin factor of `1.0`.

## 6. Pre-match and post-match semantics

For every accepted match, the ledger records:

- both pre-match ratings;
- expected home result;
- actual result and goal margin;
- base delta, margin factor, and applied delta;
- both post-match ratings;
- parameter and formula versions;
- input source and input content hash.

The schedule output uses:

- the rating immediately before a completed match for its pre-match fields;
- the latest known rating as of the run cutoff for every unplayed fixture;
- the original preseason anchor for all `elo_preseason` fields.

No row may use its own result, or a later result, to construct its pre-match
rating.

## 7. Publication isolation

The initial implementation writes only to a staging namespace:

```text
data/processed/clubelo_fallback/ENG-Premier League/2026-2027/
```

Expected artifacts:

- `match_ledger.csv`: one row per completed match;
- `team_history.csv`: anchor and post-match rating states;
- `schedule.csv`: current schedule-compatible team-side output;
- `audit.json`: counts, hashes, parameters, cutoff, and validation results.

It must not overwrite `data/raw/clubelo` or `data/processed/clubelo`. Downstream
adoption is a separate, explicit publication step after review.

Writes must be atomic: validate a complete temporary artifact set, then replace
the fallback artifact set. A failed run leaves the last valid set intact and
returns a non-zero exit code.

## 8. Operational guards

- Require an explicit `--as-of` cutoff; never infer future eligibility solely
  from the computer clock.
- Reject result rows later than `--as-of`.
- Reject a cutoff earlier than the anchor.
- Recompute from the anchor instead of incrementally editing previous output.
- Store SHA-256 hashes for the anchor input, results input, configuration, and
  generated ledger.
- Store `generated_at_utc`, `as_of_utc`, formula version, parameters, and source
  coverage in the audit.
- Make dry-run/validation the default; require an explicit publish flag to write
  artifacts.
- Never silently substitute a 1500 rating for an unknown team.
- Never silently accept only one side of a paired result.
- Never report success when any required validation fails.

## 9. Validation gates before activation

### Unit invariants

- equal teams at neutral venue produce `expected_home == 0.5`;
- expectation is symmetric when teams and venue are reversed;
- a draw between equal teams changes nothing at neutral venue;
- every match is exactly zero-sum;
- winners gain and losers lose rating points;
- an upset moves more points than the expected result in the same mode;
- replaying identical inputs produces byte-stable logical output;
- changing one result affects only that match and chronologically later states.

### Data invariants

- exactly 20 anchored teams for this season;
- exactly 380 unique league fixtures and 760 team-side schedule rows;
- every accepted result has two mirrored sides;
- every accepted result maps to a known scheduled fixture;
- no duplicate match is applied;
- completed-result count in the audit equals the ledger row count;
- total league Elo after replay equals total anchor Elo within `1e-8`.

### Historical replay

Replay at least the complete 2025-2026 Premier League season from a known
ClubElo anchor and compare generated pre-match ratings with stored ClubElo
pre-match ratings.

Report, at minimum:

- mean absolute error;
- median absolute error;
- 90th and 95th percentile absolute error;
- maximum absolute error;
- error by matchweek;
- rank correlation;
- drift after clubs play non-league matches.

The comparison selects `base` or a versioned `calibrated_margin` model. It must
not tune against 2026-2027 results.

## 10. Recovery when ClubElo returns

On recovery:

1. fetch real ClubElo data without touching fallback artifacts;
2. compare real and fallback ratings at the same timestamps;
3. publish a reconciliation audit by team and date;
4. switch downstream consumers back to provider `clubelo` explicitly;
5. retain fallback artifacts for provenance, but do not blend the two series.

The fallback is never used to backfill or rewrite authoritative ClubElo history.

## 11. Decisions fixed for the first implementation

- Existing ClubElo scraper remains unchanged.
- Premier League only.
- Preseason anchor date: `2026-08-20`.
- `k = 20`, divisor `400`, fixed `hfa = 36`.
- Safe default margin mode: `base`.
- Full recomputation from anchor on every run.
- Understat completed scores are accepted only through the mirrored-row gate.
- Separate provider label and output namespace.
- Validation/dry-run before publication.

## 12. Deliberately deferred decisions

- enabling calibrated goal-margin normalization;
- ingesting cup and European match results;
- reproducing ClubElo's dynamic country-level HFA;
- Tilt and exact-score modelling;
- automatic downstream selection of the fallback provider.

These are not required to produce a safe temporary Elo series, and none should
be added implicitly.

## 13. Commands

Validate the current season without writing any artifacts:

```powershell
python -m fpl_assistant.providers.clubelo.fallback.elo_fallback `
  --as-of 2026-09-04T23:59:59Z
```

The command fails if a scheduled date before the cutoff has no accepted result.
Use a cutoff matching the freshness of the results input; do not bypass that
gate merely to make a stale dataset pass.

Publish only after inspecting the dry-run audit:

```powershell
python -m fpl_assistant.providers.clubelo.fallback.elo_fallback `
  --as-of 2026-09-04T23:59:59Z `
  --publish
```

This writes the isolated artifact set below:

```text
data/processed/clubelo_fallback/ENG-Premier League/2026-2027/
```

Historical replay against the stored 2025-2026 ClubElo schedule:

```powershell
python -m fpl_assistant.providers.clubelo.fallback.elo_fallback `
  --season 2025-2026 `
  --anchor-schedule "data/processed/clubelo/ENG-Premier League/2025-2026/schedule.csv" `
  --results-schedule "data/processed/understat/ENG-Premier League/2025-2026/schedule.csv" `
  --reference-schedule "data/processed/clubelo/ENG-Premier League/2025-2026/schedule.csv" `
  --anchor-date 2025-08-14 `
  --as-of 2026-06-01T23:59:59Z
```

The base-mode replay currently compares all 760 team-fixture rows, with an Elo
MAE of about 11.81, a median absolute error of about 8.15, and rank correlation
of about 0.986. The remaining drift is expected because the local replay omits
cup/European matches and ClubElo's normalized goal-margin and dynamic-HFA logic.

Some historical Understat matches contain different `round` values on their two
mirrored team rows. Round does not affect Elo chronology, so these matches are
accepted with a blank generated `gw_played` and their IDs are exposed in the
audit instead of guessing a gameweek.
