# FPL Assistant data architecture

## Data flow

```text
FPL API  Understat  ClubElo  FBref  WhoScored
   |         |         |       |        |
 provider-owned raw snapshots + coverage metadata
                       |
          canonical identity and match bridges
                       |
      staged provider records (canonical IDs)
                       |
 canonical match facts + field-level provenance/conflicts
                       |
 complete player-fixture panel + as-of feature snapshots
                       |
             training, forecasts, optimizer
```

Provider raw tables are append-only inputs. Models and features must consume
canonical tables, never join provider names directly. A new provider is added
by building identity bridges and staging mappings; it does not change model
keys.

## Provider responsibilities

| Provider | Primary responsibility |
|---|---|
| FPL API | FPL IDs, fixtures/gameweeks, prices, positions, official minutes and outcomes |
| Understat | match/player xG, xA, shots and chance-quality features |
| ClubElo | pre-match team strength |
| FBref | reduced corroborating season/match tables, goalkeeping, shooting, misc, lineups and events |
| WhoScored | event-level defensive actions, formations, lineups and availability detail |

Metric selection is deterministic and stored in field-level provenance.
Defensive match metrics use WhoScored because the current FBref player-match
surface no longer provides the historical defensive tables.

## Current FBref contract

The legacy scraper is retained, but the network boundary accepts only:

- team season: `standard`, `keeper`, `shooting`, `playing_time`, `misc`
- team match: `schedule`, `keeper`, `shooting`, `misc`
- player season: `standard`, `keeper`, `shooting`, `playing_time`, `misc`
- player match: `summary`, `keepers`
- supplementary: `lineups`, `events`

Unsupported historical categories fail before scraping. Every scrape writes
`_meta/fbref_capabilities.json` and `_meta/coverage_manifest.json`. Empty
fallback files are marked `schema_only`; they are not silently treated as
successful data.

## Canonical tables

- `dim_player`, `dim_team`, `dim_match`
- `bridge_{player,team}_provider_id`, `bridge_match_provider_id`
- `fact_player_match`, `fact_team_match`
- matching `*_provenance` and `*_conflicts`
- `player_fixture_panel`, including explicit zero-minute DNP rows for completed
  fixtures and unknown values for future/postponed fixtures

Identity aliases are reviewed at ingestion. Canonical IDs are persistent and
do not depend on display names.

## Reproducible runs

The `fpl-canonical` command exposes:

- `identities`, `matches`, `stage-facts`, `facts`
- `dnp-panel`, `whoscored-defense`
- `feature-snapshot`, `run-manifest`

Feature snapshots enforce an as-of timestamp and table contract. Run manifests
record configuration, provider state, input/output hashes, code revision,
contract versions and feature versions.

Operating modes are:

- `full`: all five providers
- `standard`: FPL API, Understat and ClubElo
- `minimum`: FPL API and ClubElo

FBref or WhoScored failure can therefore degrade a run explicitly without
making the core forecast pipeline non-functional.
