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

The maintained entry points live under `fpl_assistant.providers.fpl`. The
similarly named modules under `scripts/fpl_pipeline` are legacy compatibility
copies and should not be used in new jobs.

The complete sequence is:

1. `scrape.season_scraper`: download the bootstrap roster, teams, fixtures,
   player history, and available gameweek data into the league-scoped raw tree.
2. `pipelines.clean_and_enrich`: attach official FPL context, resolve canonical
   player/team IDs, and publish the season roster.
3. `clean.gw_stats_cleaner`: clean individual and merged gameweek rows and
   attach canonical player/team IDs.
4. `clean.assign_game_ids`: attach canonical match IDs after the fixture
   calendar and FBref match surface are available.
5. `pipelines.prices_from_merged`: publish per-season price registries. Before
   GW1 it uses the processed roster price as the opening GW1 price.
6. `master.consolidate_master`: rebuild the FPL player master from enriched
   season rosters and price registries.

Steps 3 and 4 are not required for a preseason roster-only publication.

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

## 3. Clean gameweek data

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

## 4. Assign canonical game IDs

This stage requires the processed FBref season and canonical fixture calendar:

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

Skip this step in preseason or whenever the required match sources are not yet
published. Do not manufacture match IDs from names alone.

## 5. Build the price registry

```powershell
python -m fpl_assistant.providers.fpl.pipelines.prices_from_merged `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --out-json-dir "data/processed/registry/prices" `
  --out-parquet-dir "data/processed/registry/prices_parquet" `
  --season "2026-2027" `
  --log-level INFO
```

## 6. Consolidate the FPL master

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
- an unstarted season does not expose non-zero cumulative performance values;
- manual-review and carry-over audit files have been reviewed;
- gameweek rows, when present, reference the same processed season roster.

The Streamlit player history page discovers
`data/processed/fpl/<league>/<season>/season/cleaned_players.csv`; therefore a
file becomes application-visible as soon as it is published there and the
Streamlit data cache is refreshed.
