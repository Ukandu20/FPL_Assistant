# Fixture eligibility and DNP reconstruction

This pipeline builds the fixture-level population required by the hardened
expected-minutes contract. It is provider-neutral after canonical identity
resolution and does not require FBref.

## Outputs

For each season it publishes:

```text
data/processed/registry/fixtures/<season>/player_fixture_calendar.csv
data/processed/registry/fixtures/<season>/player_fixture_calendar_observed.csv
data/processed/registry/eligibility/<season>/player_eligibility.csv
data/processed/registry/eligibility/<season>/backfill_audit.json
```

`player_fixture_calendar_observed.csv` preserves the former appearance or
matchday-squad surface. `player_fixture_calendar.csv` is the expanded modelling
surface. Completed FPL fixture rows with zero minutes become explicit DNP
observations; pending fixtures retain null minutes and cannot become training
labels.

File ownership is strict: the provider calendar builder writes only the
`_observed` artifact, while this publisher is the sole writer of the expanded
modelling artifact. Team-level match context is named `team_gf`, `team_ga`,
`team_xg`, and `team_xga`. Player-level attacking columns are `goals`,
`assists`, `xg`, and `xa`, sourced from canonical Understat player-match data;
`gls` and `ast` are retained only as compatibility aliases.

The effective-dated eligibility table contains the required fields:

```text
player_id
team_id
valid_from
valid_until
registered
confirmed_unavailable
unavailable_reason
information_timestamp
source
```

It also records season, player name/position, timestamp safety, reconstruction
method, boundary match IDs, and the number of contiguous fixture observations.
Separate intervals are emitted for transfers, returns, or gaps in the archived
FPL fixture-level roster.

## Run the backfill

```powershell
python -m fpl_assistant.providers.fpl.pipelines.eligibility_backfill `
  --processed-fpl-root "data/processed/fpl/ENG-Premier League" `
  --raw-fpl-root "data/raw/fpl/ENG-Premier League" `
  --fixtures-root "data/processed/registry/fixtures" `
  --registry-root "data/processed/registry" `
  --eligibility-root "data/processed/registry/eligibility" `
  --availability-root "data/processed/registry/availability" `
  --force `
  --log-level INFO
```

Pass `--season "2026-2027"` one or more times to restrict publication.

## Provenance and leakage boundary

Historical membership reconstructed from `merged_gws.csv` is labelled
`fpl_merged_gws_retrospective` and `timestamp_safe=false`. It is valid for
reconstructing the target population and DNP labels, but its file timestamp
must never be used as a pre-deadline feature timestamp.

Provider-only appearances absent from the archived FPL GW universe are retained
with `provider_only_observation` provenance. Their eligibility is bounded to
observed evidence; the pipeline does not invent surrounding DNP rows.

For a preseason or the pending portion of an active season, the official FPL bootstrap roster is crossed with
the canonical fixture calendar. Rows use `observation_status=fixture_pending`
and null match outcomes, including minutes, bonus, BPS, points, goals, and
assists. Season-total roster values must not become future fixture results.
Price and projection context are retained separately. The minutes model loader excludes pending and explicitly
ineligible rows from training.

## Availability snapshots

Each preseason/current-season run records the source file's retrieval timestamp
and appends the official FPL status snapshot to:

```text
data/processed/registry/availability/<season>/availability_history.csv
data/processed/registry/availability/<season>/snapshots/availability__<timestamp>.csv
```

Run roster acquisition and this publisher before every prediction deadline.
Never overwrite old snapshots. A confirmed absence is usable only when its
`information_timestamp` is earlier than the relevant prediction cutoff.
Doubtful status is not treated as confirmed unavailability.

The fixture calendar is also snapshotted under:

```text
data/processed/registry/fixtures/<season>/snapshots/
```

This prevents future backtests from treating a later reschedule as information
known at an earlier deadline. Historical pre-deadline schedule snapshots cannot
be recreated when no archived source snapshot exists; those folds must carry a
schedule-leakage limitation or omit schedule-sensitive features.

## Remaining evidence limits

- Exact historical injury, suspension, loan, and registration publication
  timestamps cannot be manufactured. Historical rows without timestamped
  evidence mean “not confirmed unavailable in the archive,” not “confirmed
  healthy.”
- The 2020-2021 and 2021-2022 sources do not contain reliable complete starting
  XI labels. Those seasons now support DNP/minutes training but must not be used
  as exact `P(start)` labels without an additional lineup source. Some 2022-2023
  team-match starter counts also remain incomplete and are surfaced by
  assurance warnings.
- Live V1/V2 acceptance remains time-gated. Store both forecast versions before
  each 2026-2027 deadline, then append actual minutes and FPL points only after
  the fixture completes. Do not backfill “forecasts” from post-deadline data.
