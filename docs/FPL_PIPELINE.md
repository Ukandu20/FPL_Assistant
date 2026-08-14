# FPL pipeline runbook

This runbook is the production contract for scraping, cleaning, enriching, and
publishing Fantasy Premier League data.

## Storage contract

FPL data is always league-scoped:

```text
data/raw/fpl/<league>/<season>/
data/processed/fpl/<league>/<season>/
```

For the Premier League, `<league>` is `ENG-Premier League` and `<season>` uses
the long form `YYYY-YYYY`, for example `2026-2027`.

The command-line tools accept either the provider root (`data/raw/fpl`) or the
already scoped root (`data/raw/fpl/ENG-Premier League`). They add the league
folder only when needed. New automation should use the explicit league-scoped
paths shown below.

Raw files are source snapshots. Applications and models must read processed
files only.

## Complete cleaning suite

The maintained FPL entry points live under `fpl_assistant.providers.fpl`. The
similarly named modules under `scripts/fpl_pipeline` are legacy compatibility
copies and should not be used in new jobs. Canonical fixture bootstrapping is a
cross-provider integration command and therefore lives under
`fpl_assistant.providers.fbref.integrate` despite reading only FPL inputs in
bootstrap mode.

The complete sequence is:

1. `scrape.season_scraper`: download the bootstrap roster, teams, fixtures,
   player history, and available gameweek data into the league-scoped raw tree.
2. `pipelines.clean_and_enrich`: attach official FPL context, resolve canonical
   player/team IDs, and publish the season roster.
3. `fbref.integrate.fixtures_meta_builder --bootstrap`: publish the canonical
   fixture calendar directly from the official FPL teams and fixtures. This
   mode does not require cleaned FBref, WhoScored, or Understat data.
4. `clean.gw_stats_cleaner`: clean individual and merged gameweek rows and
   attach canonical player/team IDs.
5. `clean.assign_game_ids`: attach canonical match IDs to played gameweek rows
   after the fixture calendar and FBref match surface are available.
6. `pipelines.prices_from_merged`: publish per-season price registries. Before
   GW1 it uses the processed roster price as the opening GW1 price.
7. `master.consolidate_master`: rebuild the FPL player master from enriched
   season rosters and price registries.

Steps 4 and 5 are not required for a preseason roster-only publication. Step 3
is the supported way to make the new season's canonical fixture identities
available before provider match data exists.

## 1. Scrape a season

```powershell
python -m fpl_assistant.providers.fpl.scrape.season_scraper `
  --raw-root "data/raw/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --season "2026-2027"
```

Use `--fresh` only when intentionally replacing the generated files for that
season. The scraper writes:

```text
data/raw/fpl/ENG-Premier League/2026-2027/
  players_raw.csv
  players/
  gws/
  season/
    cleaned_players.csv
    fixtures.csv
    teams.csv
    fixture_metadata*.csv
```

## 2. Enrich and publish the season roster

```powershell
python -m fpl_assistant.providers.fpl.pipelines.clean_and_enrich `
  --raw-root "data/raw/fpl/ENG-Premier League" `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --fbref-master "data/processed/registry/master_players.json" `
  --overrides "data/processed/registry/overrides.json" `
  --team-map "data/processed/registry/_id_lookup_teams.json" `
  --generate-missing-ids `
  --season "2026-2027" `
  --log-level INFO
```

The published roster is:

```text
data/processed/fpl/ENG-Premier League/2026-2027/season/cleaned_players.csv
```

### New-player identity registration

With `--generate-missing-ids`, the cleaner first checks the canonical player
master, name lookup, manual overrides, FPL provider bridges, and prior FPL
season rosters. An approved provider bridge is authoritative. A historical
generated provider-code identity is only a fallback when the player still has
no established canonical name identity.

For a genuinely new player, the fallback ID is the deterministic 8-character
hash of the stable FPL `code`. The cleaner validates that the resulting ID,
canonicalized name, and provider code do not belong to different players. It
fails the run on a collision rather than silently adding a duplicate identity.
An ID already used for the same FPL code in an earlier season is retained for
historical join stability, even if that legacy ID predates the 8-character
policy.

Registration is enabled by default and atomically promotes each generated
identity into all maintained machine registries:

```text
data/processed/registry/master_players.json
data/processed/registry/_id_lookup_players.json
data/processed/registry/master_fpl.json
data/processed/registry/bridges/player_ids.csv
data/processed/fpl/ENG-Premier League/master_fpl_players.json
```

The manual overrides file is not modified. It remains a reviewed input, not a
generated registry. The season audit files are:

```text
data/processed/fpl/ENG-Premier League/<season>/_manual_review/
  generated_ids_<season>.csv
  registered_generated_ids_<season>.csv
```

Use `--no-register-generated-players` only for a diagnostic dry publication
that must not update the registries. Re-running normal registration is
idempotent and does not duplicate FPL bridge rows.

### Player-season metric enrichment

Roster publication also attaches normalized provider metrics without changing
the input row count. Provider routing is deterministic:

| Seasons | Defense and DEFCON | xG/xA | Goalkeepers |
|---|---|---|---|
| through 2024-2025 | FBref player-season defense | FBref player-season standard (`xag` becomes `xa`) | FBref player-season keeper |
| 2025-2026 onward | WhoScored player-season defense | Understat player-season | WhoScored player-match keeper appearances |

`defcon` is the project-defined sum of `blocks + interceptions + clearances`.
It remains null unless all three ingredients are available. Goalkeeper fields
include shots on target against, saves, goals against, save percentage,
penalties faced, and the historical FBref penalty outcome splits. Modern
WhoScored goalkeeper totals are rebuilt from rows with `fpl_pos=GKP` and
positive minutes because the broad player-season keeper table also contains
bench and outfield roster rows.

Every run writes field provenance, coverage status, unmatched active players,
row-count verification, and shared-goalkeeper-match warnings to:

```text
data/processed/fpl/ENG-Premier League/<season>/_manual_review/
  player_season_stat_enrichment_<season>.json
```

To backfill only these metrics into existing published rosters:

```powershell
python -m fpl_assistant.providers.fpl.pipelines.clean_and_enrich `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --fbref-root "data/processed/fbref" `
  --whoscored-root "data/processed/whoscored" `
  --understat-root "data/processed/understat" `
  --stats-only `
  --season all
```

The backfill refuses duplicate provider join keys rather than allowing a merge
to multiply roster rows. Missing future-season provider files produce null
metrics with `provider_data_unavailable` coverage instead of fabricated zeros.

### Preseason carry-over protection

If every target-season fixture is unstarted but the raw roster contains
non-zero cumulative performance values, the cleaner treats those values as a
prior-season carry-over. It retains current roster fields such as player,
team, position, price, status, and selection percentage, but resets cumulative
fields such as minutes, points, goals, assists, bonus, BPS, cards, and ICT
components to zero.

The processed rows are marked with:

```text
season_data_status=preseason_roster
performance_data_status=prior_season_carryover_reset
```

An audit is written to:

```text
data/processed/fpl/ENG-Premier League/<season>/_manual_review/
  preseason_carryover_reset_<season>.json
```

This is the required treatment for the current 2026-2027 snapshot: its roster
and fixtures belong to 2026-2027, but its cumulative player values belong to
the preceding season and must not be displayed as 2026-2027 results.

## 3. Bootstrap the canonical fixture calendar

Run this immediately after the official FPL teams and fixtures have been
scraped. Bootstrap mode reads only FPL inputs; it does not depend on cleaned
WhoScored, Understat, or FBref schedules.

```powershell
python -m fpl_assistant.providers.fbref.integrate.fixtures_meta_builder `
  --bootstrap `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fpl-root "data/raw/fpl/ENG-Premier League" `
  --team-map "data/processed/registry/_id_lookup_teams.json" `
  --short-map "data/config/teams.json" `
  --out-dir "data/processed/registry/fixtures" `
  --force `
  --log-level INFO
```

Expected output:

```text
data/processed/registry/fixtures/2026-2027/fixture_calendar.csv
```

The calendar contains two team-perspective rows for every FPL fixture. Its
canonical `match_id` is stable across kickoff changes because it is derived
from league, season, home team, and away team rather than the scheduled date.
The `fbref_id` column is retained as a compatibility alias in bootstrap output;
it does not mean that FBref supplied the identity.

### Preseason downstream provider hand-off

Once steps 1-3 are complete, the supported preseason dependency order is:

```text
FPL scrape and clean
  -> FPL-only fixture bootstrap
  -> WhoScored partial clean
  -> ClubElo fixture schedule
```

WhoScored can now resolve its scraped schedule against the bootstrap calendar
even when no match events exist:

```powershell
python -m fpl_assistant.providers.whoscored.clean.whoscored_cleaner `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --allow-partial `
  --force `
  --log-level INFO
```

This publishes the provider-owned schedule with explicit `partial` coverage.
It also always publishes:

```text
data/processed/whoscored/ENG-Premier League/2026-2027/player_season/roles.csv
```

`roles.csv` is strictly season-specific and is never seeded from the preceding
season. It derives the observed `corner`, `penalty`, `direct_free_kick`,
`free_kick`, `indirect_free_kick`, and `long_throw` hierarchies from WhoScored
events. Corner rows keep `role=corner`; their `side` is `left` or `right`.
Before the first 2026-2027 event is available, the file is intentionally
schema-valid and has zero rows, including for promoted teams.

Publish the ClubElo-owned fixture schedule from the same canonical calendar:

```powershell
python -m fpl_assistant.providers.clubelo.clean.clubelo_understat_enricher `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fixture-root "data/processed/registry/fixtures"
```

This writes
`data/processed/clubelo/ENG-Premier League/2026-2027/schedule.csv` with dynamic
pre-match Elo fields and season-frozen preseason Elo fields.

## 4. Clean gameweek data

Run this after at least one gameweek file exists:

```powershell
python -m fpl_assistant.providers.fpl.clean.gw_stats_cleaner `
  --raw-root "data/raw/fpl/ENG-Premier League" `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --master "data/processed/registry/master_players.json" `
  --overrides "data/processed/registry/overrides.json" `
  --team-map "data/processed/registry/_id_lookup_teams.json" `
  --short-map "data/config/teams.json" `
  --season "2026-2027" `
  --on-unmatched keep `
  --log-level INFO
```

The cleaner reads `season/teams.csv` and writes cleaned GW files beneath the
same league and season in the processed tree. Review unmatched-player outputs
before using `--on-unmatched drop` in a production build.

## 5. Assign canonical game IDs

The bootstrap calendar already contains canonical match identities, but this
stage attaches them to played FPL gameweek rows. The current command also uses
the processed FBref match surface as a fallback and validation source, so run
it only after both the merged FPL gameweeks and FBref summary are available:

```powershell
python -m fpl_assistant.providers.fpl.clean.assign_game_ids `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --fbref-root "data/processed/fbref" `
  --fixture-calendar-root "data/processed/registry/fixtures" `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --tz UTC `
  --log-level INFO
```

Skip this step in preseason or whenever the required played-match sources are
not yet published. Do not manufacture match IDs from names alone.

## 6. Build the price registry

```powershell
python -m fpl_assistant.providers.fpl.pipelines.prices_from_merged `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --out-json-dir "data/processed/registry/prices" `
  --out-parquet-dir "data/processed/registry/prices_parquet" `
  --season "2026-2027" `
  --log-level INFO
```

## 7. Consolidate the FPL master

```powershell
python -m fpl_assistant.providers.fpl.master.consolidate_master `
  --fbref-master "data/processed/registry/master_players.json" `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --prices-dir "data/processed/registry/prices" `
  --out-json "data/processed/registry/master_fpl.json" `
  --league "ENG-Premier League" `
  --season latest `
  --log-level INFO
```

## Validation checklist

Before allowing an application to discover a new processed season, verify:

- the raw and processed paths contain the league folder exactly once;
- `season/fixtures.csv` contains kickoff dates within the target season;
- the processed roster has non-null, unique `player_id` values;
- every player has `team_id`, `fpl_pos`, and `fpl_element_id`;
- genuinely new generated players use 8-character IDs and appear in every
  maintained player registry and the FPL provider bridge;
- `generated_ids_<season>.csv` and
  `registered_generated_ids_<season>.csv` contain the same promoted identities;
- an unstarted season does not expose non-zero cumulative performance values;
- the bootstrap fixture calendar contains two rows per fixture, unique
  `(match_id, team_id)` keys, and no missing canonical team IDs;
- an eventless preseason has a schema-valid, zero-row WhoScored `roles.csv`;
- the ClubElo schedule has complete preseason and pre-match Elo coverage;
- manual-review and carry-over audit files have been reviewed;
- gameweek rows, when present, reference the same processed season roster.

The Streamlit player history page discovers
`data/processed/fpl/<league>/<season>/season/cleaned_players.csv`; therefore a
file becomes application-visible as soon as it is published there and the
Streamlit data cache is refreshed.
