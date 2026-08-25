# Provider integration pipeline runbook

This is the production runbook for building canonical fixture, team-form, and
player-form artifacts. FBref is optional: current-season publication can use
official FPL schedules, WhoScored match data, and Understat expected metrics.
Commands are PowerShell commands and assume they run from the repository root.

## Current 2026-2027 source boundary

The provider-neutral path is the default for 2026-2027:

| Artifact | Required source | Optional enrichment |
|---|---|---|
| canonical fixture bootstrap | FPL fixtures and teams | none |
| completed fixture calendar | FPL + WhoScored + Understat | FBref schedule validation |
| FPL canonical match IDs | fixture calendar | historical FBref fallback |
| player fixture calendar | fixture calendar + WhoScored | none |
| team/player form | canonical calendars | none |
| assurance | canonical calendars + registries | FBref lineup cross-check |

The pipeline must not stop merely because `data/raw/fbref` or
`data/processed/fbref` has no 2026-2027 folder. Canonical `match_id` is the
provider-neutral identity. `fbref_id` remains in published files as a temporary
compatibility alias and must not be interpreted as proof of FBref provenance.

## Optional historical FBref acquisition

FBref removed the advanced match-report tables used by the historical
pipeline. Their absence is a provider limitation, not a cleaning error. For
seasons where FBref is acquired, the active processed dataset must use only tables that FBref still
publishes completely:

| Family | Accepted optional FBref tables |
|---|---|
| `team_season` | `standard`, `keeper`, `shooting`, `playing_time`, `misc` |
| `player_season` | `standard`, `keeper` |
| `player_match` | `schedule`, `summary`, `keepers` |
| `team_match` | none; advanced partial outputs are quarantined |

Do not interpret a schema-only or truncated advanced table as a valid zero-row
dataset. WhoScored supplies event and tactical detail, and Understat supplies
the chance-quality metrics that are no longer available from FBref.

## 1. Prepare the environment

Use the project environment in which `fpl_assistant` and its dependencies are
installed. UTF-8 avoids Windows console failures. Keeping soccerdata's cache
inside the project also makes cache state easier to inspect.

```powershell
$env:PYTHONUTF8 = "1"
$env:SOCCERDATA_DIR = (Resolve-Path "data").Path + "\_soccerdata"
$env:SOCCERDATA_LOGLEVEL = "ERROR"
```

The FBref cleaner needs the already-cleaned official FPL roster at:

```text
data/processed/fpl/ENG-Premier League/2026-2027/
```

That dependency is intentional. FPL is authoritative for `fpl_pos`, while
FBref's provider position remains available as tactical detail.

## 2. Optional: acquire or resume an FBref snapshot

Skip this entire section for an FBref-free run. When deliberately acquiring
FBref, the snapshot belongs under
`data/raw/fbref/ENG-Premier League/2026-2027/`.

For a deliberately fresh 2026-2027 season-level pull:

```powershell
python -m fpl_assistant.providers.fbref.scrape.season_stats_scraper `
  --league "ENG-Premier League" `
  --seasons "2026-2027" `
  --out-dir "data/raw/fbref" `
  --levels both `
  --team-mode direct `
  --player-stats standard keeper `
  --team-stats standard keeper shooting playing_time misc `
  --refresh `
  --browser-path "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --headless `
  --verbose
```

For the supported match-level surface, request player match reports only. The
bare `--supplementary-stats` option intentionally selects no supplementary
tables for this season:

```powershell
python -m fpl_assistant.providers.fbref.scrape.match_stats_scraper `
  --league "ENG-Premier League" `
  --seasons "2026-2027" `
  --out-dir "data/raw/fbref" `
  --levels player `
  --player-stats summary keepers `
  --supplementary-stats `
  --refresh `
  --browser-path "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --headless `
  --verbose
```

If Chrome is installed elsewhere, change `--browser-path` or omit it and let
the browser transport discover Chrome. Do not point this option at
`msedge.exe`; the current transport validates Chromium executables by filename
and rejects it.

### Cache behavior after an interrupted scrape

An interrupted run does not require a cache bypass. Successful responses
already in soccerdata's cache remain usable. Resume without `--refresh` or
`--no-cache`, and skip complete outputs:

```powershell
python -m fpl_assistant.providers.fbref.scrape.match_stats_scraper `
  --league "ENG-Premier League" `
  --seasons "2026-2027" `
  --out-dir "data/raw/fbref" `
  --levels player `
  --player-stats summary keepers `
  --supplementary-stats `
  --skip-existing `
  --skip-schedule `
  --headless `
  --verbose
```

Use `--rerun-failed` when `data/meta/scraper_runs.json` contains a recorded
partial run. Use `--force-cache` only for an intentionally offline run. Use
`--refresh`/`--no-cache` only when the cached response itself is stale or bad.

## 3. Optional: inspect FBref coverage before cleaning

Review these files when present:

```text
data/raw/fbref/ENG-Premier League/2026-2027/_meta/fbref_capabilities.json
data/raw/fbref/ENG-Premier League/2026-2027/_meta/coverage_manifest.json
data/meta/scraper_runs.json
```

Do not continue if `schedule.csv` has no in-season match dates, a required
table is marked `schema_only`, or a table contains only headers. The cleaner
also enforces the schedule and schema-only guards, but checking the manifest
makes the cause of a rejection clearer.

## 4. Optional: run the FBref cleaner

The league-scoped `--fpl-root` is required when cleaning FBref. Passing the older
`data/processed/fpl` root prevents the cleaner from finding the official FPL
positions.

```powershell
python -m fpl_assistant.providers.fbref.clean.csv_cleaner `
  --raw-dir "data/raw/fbref" `
  --clean-dir "data/processed" `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fpl-root "data/processed/fpl/ENG-Premier League" `
  --force `
  --log-level INFO
```

The cleaner performs the following work sequentially:

1. validates the raw season and schedule;
2. normalizes headers, nulls, numeric values, dates, team names, and positions;
3. resolves canonical team, player, and match identities;
4. retains native IDs in provider-specific ID columns;
5. applies official FPL position as `fpl_pos` without deleting FBref tactical
   position fields;
6. writes provider-owned tables below
   `data/processed/fbref/ENG-Premier League/2026-2027/`;
7. updates the identity lookup/audit artifacts only after valid input passes
   the safety checks.

Cleaning is deliberately single-threaded because it mutates shared identity
registries. The accepted `--workers` option is retained only for CLI
compatibility.

## 5. Optional: quarantine deprecated FBref partial outputs

For 2026-2027, any generated `team_match/keeper.csv`,
`team_match/shooting.csv`, or `team_match/misc.csv` is not production data.
These explicit commands preserve the files for audit while removing them from
the active processed tree:

```powershell
$fbrefActive = "data/processed/fbref/ENG-Premier League/2026-2027/team_match"
$fbrefQuarantine = "data/quarantine/fbref/ENG-Premier League/2026-2027/incomplete_processed/team_match"
New-Item -ItemType Directory -Force -Path $fbrefQuarantine | Out-Null

if (Test-Path "$fbrefActive/keeper.csv") {
  Move-Item -Force "$fbrefActive/keeper.csv" "$fbrefQuarantine/keeper.csv"
}
if (Test-Path "$fbrefActive/shooting.csv") {
  Move-Item -Force "$fbrefActive/shooting.csv" "$fbrefQuarantine/shooting.csv"
}
if (Test-Path "$fbrefActive/misc.csv") {
  Move-Item -Force "$fbrefActive/misc.csv" "$fbrefQuarantine/misc.csv"
}
```

Raw snapshots remain immutable. Because the cleaner scans all raw CSVs, repeat
this quarantine step after a forced clean if an older raw snapshot still
contains those deprecated partial files.

## 6. Optional: validate cleaned FBref tables

Check the active surface and basic Premier League cardinalities:

```powershell
$fbrefSeason = "data/processed/fbref/ENG-Premier League/2026-2027"
$schedule = Import-Csv "$fbrefSeason/player_match/schedule.csv"
$summary = Import-Csv "$fbrefSeason/player_match/summary.csv"

if (($schedule.game_id | Sort-Object -Unique).Count -ne 380) {
  throw "FBref schedule does not contain 380 unique matches"
}
if (($summary | Where-Object { -not $_.game_id -or -not $_.player_id }).Count) {
  throw "FBref summary contains unresolved game_id or player_id values"
}
if (Test-Path "$fbrefSeason/team_match/keeper.csv") {
  throw "Deprecated team-match output is still active"
}
if (Test-Path "$fbrefSeason/team_match/shooting.csv") {
  throw "Deprecated team-match output is still active"
}
if (Test-Path "$fbrefSeason/team_match/misc.csv") {
  throw "Deprecated team-match output is still active"
}
```

Also review cleaner warnings and identity audits. A clean command exiting zero
does not make a source-deprecated table complete; publication is governed by
the accepted surface above.

## 7. Build the canonical fixture calendar

For a new season, first bootstrap the calendar from FPL alone. This can run
before WhoScored, Understat, or FBref has published match data and gives their
cleaners a stable canonical `match_id` to resolve against.

```powershell
python -m fpl_assistant.providers.fbref.integrate.fixtures_meta_builder `
  --bootstrap `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fpl-root "data/raw/fpl/ENG-Premier League" `
  --out-dir "data/processed/registry/fixtures" `
  --force `
  --log-level INFO
```

After scraping Understat, clean the raw season before building the enriched
calendar. This produces the required processed `schedule.csv` and applies the
league-scoped FPL player aliases.

```powershell
python -m fpl_assistant.providers.understat.clean.clean_understat_raw `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --in-root "data/raw/understat" `
  --out-root "data/processed/understat" `
  --fpl-root "data/processed/fpl/ENG-Premier League" `
  --verbose
```

After WhoScored and Understat publish completed-match data, build the enriched
calendar without FDR. FPL supplies fixture IDs, gameweeks, and the scheduled
calendar. FBref is consulted only when its optional schedule exists.

```powershell
python -m fpl_assistant.providers.fbref.integrate.fixtures_meta_builder `
  --season "2026-2027" `
  --fpl-root "data/raw/fpl/ENG-Premier League" `
  --whoscored-league-dir "data/processed/whoscored/ENG-Premier League" `
  --understat-league-dir "data/processed/understat/ENG-Premier League" `
  --team-map "data/processed/registry/_id_lookup_teams.json" `
  --short-map "data/config/teams.json" `
  --out-dir "data/processed/registry/fixtures" `
  --features-root "data/processed/registry/features" `
  --force `
  --log-level INFO
```

Expected output:

```text
data/processed/registry/fixtures/2026-2027/fixture_calendar.csv
```

It should contain 760 team-fixture rows, representing both sides of 380
matches. Review `_reschedule_audit.csv`; moved fixtures are expected when the
FPL schedule changed, but unresolved matches are not.

`venue` retains WhoScored's physical stadium name. Team perspective is stored
separately as `is_home` in the fixture calendar and `was_home` in the player
calendar; downstream form calculations use those flags rather than parsing the
stadium string.

## 8. Assign canonical match IDs to FPL rows

```powershell
python -m fpl_assistant.providers.fpl.clean.assign_game_ids `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --fixture-calendar-root "data/processed/registry/fixtures" `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --tz UTC `
  --log-level INFO
```

This updates cleaned FPL gameweek rows with provider-neutral `match_id` and the
legacy `game_id` alias. If an optional historical FBref summary exists it is a
fallback, not a prerequisite. FPL's official fixture ID and corrected
gameweek remain separate fields.

## 9. Build team form and optional FDR view

```powershell
python -m fpl_assistant.providers.fbref.integrate.team_form_builder `
  --season "2026-2027" `
  --fixtures-root "data/processed/registry/fixtures" `
  --out-dir "data/processed/registry/features" `
  --write-latest `
  --strict-ids `
  --force `
  --log-level INFO
```

After the team-form version exists, rerun the fixture builder with the same
arguments as step 7 plus `--attach-fdr latest` if a materialized fixture/FDR
view is needed.

## 10. Build the player-fixture calendar

```powershell
python -m fpl_assistant.providers.fbref.integrate.calendar_builder `
  --fixtures-root "data/processed/registry/fixtures" `
  --whoscored-root "data/processed/whoscored/ENG-Premier League" `
  --fpl-root "data/processed/fpl/ENG-Premier League" `
  --features-root "data/processed/registry/features" `
  --team-version latest `
  --season "2026-2027" `
  --force `
  --log-level INFO
```

Do not use `--create-empty` for a production run; that switch is only a
diagnostic escape hatch.

## 11. Build player form

```powershell
python -m fpl_assistant.providers.fbref.integrate.player_form_builder `
  --season "2026-2027" `
  --fixtures-root "data/processed/registry/fixtures" `
  --out-dir "data/processed/registry/features" `
  --write-latest `
  --force `
  --log-level INFO
```

Routine runs write directly to `features/latest/<SEASON>/` and do not create a
new version directory. Add `--auto-version` only when intentionally publishing
a new immutable `vN` snapshot; that snapshot's component files are also copied
into the composite `latest` directory.

## 12. Run final assurance and tests

```powershell
python -m fpl_assistant.qa.assurance `
  --fixtures-root "data/processed/registry/fixtures" `
  --teams-lookup "data/processed/registry/_id_lookup_teams.json" `
  --players-lookup "data/processed/registry/_id_lookup_players.json" `
  --seasons "2026-2027" `
  --log-level INFO

python -m pytest -q
```

Assurance is an integration check, so run it only after the fixture and player
calendars exist. The FBref lineup cross-check runs automatically when an
optional FBref lineup file exists and is skipped otherwise. Investigate
failures rather than lowering coverage thresholds for production publication.
