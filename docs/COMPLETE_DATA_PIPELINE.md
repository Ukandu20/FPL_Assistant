# Complete FPL and provider data pipeline

This is the combined execution runbook for the FPL pipeline and the provider
integration pipeline formerly described as the "FBref pipeline". It covers
scraping, identity publication, cleaning, prices, calendars, eligibility, Elo,
and team/player form through final assurance. Run the numbered stages in order;
FBref acquisition and preview data are explicitly marked optional. This runbook
does not train forecasting models or start the application.

Continue with [Archetyping and modelling](ARCHETYPING_AND_MODELLING.md) for
post-processing methodology, archetype publication, model training, forecasting,
and validation.

Examples target `ENG-Premier League`, season `2026-2027`. Understat uses `EPL`
and start year `2026`; WhoScored stores this season in raw folder `2627`.
Change these together for another season. Keep existing canonical registries,
team mappings, and reviewed overrides: this is a pipeline for the configured
repository, not a replacement for initial identity curation.

Use one PowerShell session in `C:\dev\FPL_Assistant`. Run each block separately
and stop to investigate a failed command before continuing. Do not paste `>>`
prompts. Continuation backticks must be the final character on their lines.
Cleaning and registry publication must run sequentially because they share
identity registries. Applications and models consume processed data.

## Execution order

| Stage | Produces or refreshes | Depends on |
|---|---|---|
| 1 | Environment | Project checkout |
| 2?4 | FPL raw data, canonical roster, bootstrap fixtures | Existing identity configuration |
| 5 | Cleaned FPL gameweeks | FPL roster |
| 6?7 | WhoScored and Understat raw/clean data | Roster and bootstrap fixtures |
| 8 | Optional FBref raw/clean data | Roster and bootstrap fixtures |
| 9?11 | FPL provider metrics, prices, FPL master | Cleaned providers and gameweeks |
| 12?13 | Enriched fixture calendar and FPL match IDs | Cleaned providers and FPL |
| 14 | ClubElo histories and fixture ratings | Canonical calendar and Understat |
| 15?16 | Team form and observed player calendar | Enriched fixtures and providers |
| 17?18 | Expanded eligibility/DNP calendar and player form | Observed calendar, roster, FPL, Understat |
| 19 | Assurance | Final calendars and features |

The FPL roster is published twice for different purposes: stage 3 establishes
identities before provider cleaning; stage 9 refreshes metrics after that
cleaning. Likewise, stage 4 bootstraps fixture identities and stage 12 enriches
them. Do not run the bootstrap again after enrichment within the same build.

## 1. Prepare the environment

Use the project's Python environment. Install dependencies once when setting
up that environment:

```powershell
python -m pip install -e ".[scraping,dev]"
```

Then configure the session:

```powershell
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path "data" | Out-Null
$env:SOCCERDATA_DIR = (Resolve-Path "data").Path + "\_soccerdata"
$env:SOCCERDATA_LOGLEVEL = "ERROR"
```

Keep the same cache location between runs. Changing `SOCCERDATA_DIR` makes an
existing cache elsewhere invisible to these commands.

## 2. Scrape official FPL data

```powershell
python -m fpl_assistant.providers.fpl.scrape.season_scraper `
  --raw-root "data/raw/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --season "2026-2027"
```

This fetches the roster, teams, fixtures, player histories, and available
GW data into `data/raw/fpl/ENG-Premier League/2026-2027/`. Use `--fresh` only
when intentionally replacing generated raw files for the season.

## 3. Clean and publish the FPL roster and identities

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

This publishes `season/cleaned_players.csv` in the processed FPL season and
upserts canonical player/team registries and FPL identity bridges. It must run
before provider cleaners so newly transferred or registered players can resolve.
Missing provider metrics at this point are expected; stage 9 fills them after
provider cleaning. Review `_manual_review/` identity audits; do not bypass
identity collisions. Reviewed `overrides.json` is an input, not generated output.

## 4. Bootstrap canonical fixtures from FPL

```powershell
python -m fpl_assistant.pipelines.integrate.fixtures_meta_builder `
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

Output: `data/processed/registry/fixtures/2026-2027/fixture_calendar.csv`.
It has two team-perspective rows per official fixture. Canonical `match_id`
does not change when kickoff is rescheduled; legacy `fbref_id` is a compatibility
alias and does not imply FBref provenance.

## 5. Clean FPL gameweek data

Run when raw gameweek files exist. Skip for a roster-only preseason build.

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

This publishes cleaned gameweeks and `gws/merged_gws.csv`. Review unmatched
players before downstream use. Match IDs are attached in stage 13.

## 6. Scrape and clean WhoScored

### 6a. Discover the schedule

Save the schedule before selecting completed matches.

```powershell
python -m fpl_assistant.providers.whoscored.scrape.whoscored_match_stats_scraper `
  --backend native `
  --league "ENG-Premier League" `
  --seasons "2026-2027" `
  --out-dir "data/raw/whoscored" `
  --tables schedule `
  --browser-fallback `
  --delay 0.75 `
  --headless `
  --no-cache `
  --meta-path "data/meta/scraper_runs.json" `
  --run-mode manual `
  --verbose
```

### 6b. Fetch completed-match events and statistics

Use the soccerdata backend from the existing event workflow. Do not run this
block before any matches are completed: `--completed-only` requires a saved
schedule containing completed fixtures.

```powershell
python -m fpl_assistant.providers.whoscored.scrape.whoscored_match_stats_scraper `
  --backend soccerdata `
  --league "ENG-Premier League" `
  --seasons "2026-2027" `
  --out-dir "data/raw/whoscored" `
  --tables events `
  --completed-only `
  --events-format events `
  --derived-tables match_info incidents player_dictionary lineups formations `
  --archive-raw-events `
  --raw-match-dir-layout per-match `
  --stats-mode all-visible `
  --headed `
  --retry-missing `
  --on-error raise `
  --meta-path "data/meta/scraper_runs.json" `
  --run-mode manual `
  --verbose
```

Raw output is under `data/raw/whoscored/WhoScored/ENG-Premier League/2627/`,
including `events/`, `derived/`, `stats/`, and archived per-match payloads.
A Selenium connection-reset or provider-block error is a failed acquisition,
not evidence of empty match data. Resolve it before treating this stage as
complete. A visible Chrome window does not guarantee a successful scrape.

### 6c. Optional: fetch missing-player previews separately

Previews are independent of event acquisition; a blocked preview must not
prevent attempting events. Run after schedule discovery if injury/suspension
preview data is needed.

```powershell
python -m fpl_assistant.providers.whoscored.scrape.whoscored_match_stats_scraper `
  --backend native `
  --league "ENG-Premier League" `
  --seasons "2026-2027" `
  --out-dir "data/raw/whoscored" `
  --tables missing_players `
  --browser-fallback `
  --delay 0.75 `
  --headless `
  --no-cache `
  --meta-path "data/meta/scraper_runs.json" `
  --run-mode manual `
  --verbose
```

The native fallback reads Chrome's page source immediately and closes Chrome.
`--headed` alone does not add an interactive challenge wait. Native preview
requests are not retried by `--retry-missing`.

### 6d. Clean WhoScored and rebuild provider bridges

```powershell
python -m fpl_assistant.providers.whoscored.clean.whoscored_cleaner `
  --raw-root "data/raw/whoscored" `
  --out-root "data/processed/whoscored" `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --registry-root "data/processed/registry" `
  --rebuild-provider-bridges `
  --allow-partial `
  --force `
  --log-level INFO
```

`--allow-partial` allows an in-progress season while retaining strict identity
validation. It does not certify that all completed fixtures were downloaded.
Review coverage and identity audits. Resolve aliases or refresh the FPL roster
when players are unresolved; do not use `--no-strict-identities` for publication.
The cleaner also publishes season-specific `player_season/roles.csv`; before
any events exist it should be schema-valid with zero rows.

## 7. Scrape and clean Understat

### 7a. Download expected metrics and match data

```powershell
python -m fpl_assistant.providers.understat.scrape.understat_stats_scraper `
  --league EPL `
  --seasons 2026 `
  --no-cache `
  --verbose
```

`--no-cache` refreshes the active season. Omit it when deliberately reusing a
valid cached snapshot. Raw output defaults to `data/raw/understat`.

### 7b. Clean against the published FPL roster

```powershell
python -m fpl_assistant.providers.understat.clean.clean_understat_raw `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --in-root "data/raw/understat" `
  --out-root "data/processed/understat" `
  --fpl-root "data/processed/fpl/ENG-Premier League" `
  --verbose
```

Output is under `data/processed/understat/ENG-Premier League/2026-2027/`,
including the processed schedule and player-match/season expected metrics.

## 8. Optional: acquire and clean FBref

Skip this stage for an FBref-free build. Current canonical fixtures and FPL
match-ID assignment do not require FBref. Only publish the supported surfaces:
team-season `standard`, `keeper`, `shooting`, `playing_time`, `misc`;
player-season `standard`, `keeper`; player-match `schedule`, `summary`, `keepers`.
Unavailable advanced tables must not be interpreted as zero performance.

### 8a. Scrape season and match tables

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

Change or omit `--browser-path` if Chrome lives elsewhere. For an interrupted
run, resume without `--refresh`; use `--skip-existing`, and use `--skip-schedule`
only if a valid schedule is already saved. `--force-cache` is for offline use.

### 8b. Check coverage and clean

Review the raw season's `_meta/fbref_capabilities.json`,
`_meta/coverage_manifest.json`, and `data/meta/scraper_runs.json`. Stop if
required tables are schema-only or the schedule has no in-season dates.

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

### 8c. Quarantine deprecated partial outputs and validate

If an older raw snapshot caused the cleaner to republish incomplete advanced
team-match files, preserve them outside the active processed tree:

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

The 380-match check assumes a complete Premier League fixture schedule;
it is not a requirement for 380 completed match reports during an active season.
Review cleaner identity audits as well as command exit status.

## 9. Refresh FPL metrics from the cleaned providers

This second pass only enriches the already-published roster metrics.

```powershell
python -m fpl_assistant.providers.fpl.pipelines.clean_and_enrich `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --fbref-root "data/processed/fbref" `
  --whoscored-root "data/processed/whoscored" `
  --understat-root "data/processed/understat" `
  --stats-only `
  --season "2026-2027"
```

For 2025-2026 onward, defense/keeper metrics come from WhoScored and xG/xA
from Understat. Earlier seasons use historical FBref surfaces. Review
`_manual_review/player_season_stat_enrichment_2026-2027.json` in the FPL season
for coverage, unmatched players, and row-count preservation. Missing metrics
remain null rather than becoming fabricated zeros.

## 10. Publish the FPL price registry

```powershell
python -m fpl_assistant.providers.fpl.pipelines.prices_from_merged `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --league "ENG-Premier League" `
  --out-json-dir "data/processed/registry/prices" `
  --out-parquet-dir "data/processed/registry/prices_parquet" `
  --season "2026-2027" `
  --log-level INFO
```

Before GW1 this uses the processed roster price as the opening GW1 price.

## 11. Consolidate the FPL master

```powershell
python -m fpl_assistant.providers.fpl.master.consolidate_master `
  --fbref-master "data/processed/registry/master_players.json" `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --prices-dir "data/processed/registry/prices" `
  --out-json "data/processed/registry/master_fpl.json" `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --log-level INFO
```

## 12. Enrich the canonical fixture calendar

Run after WhoScored and Understat cleaning. This adds provider match context
to the FPL fixture calendar. Leave FDR unattached until team form is published.

```powershell
python -m fpl_assistant.pipelines.integrate.fixtures_meta_builder `
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

Review `_reschedule_audit.csv` in the fixture output. A complete 380-fixture
calendar has 760 team rows and unique `(match_id, team_id)` keys. Moved dates
are expected when official fixtures change; unresolved identities are not.
The physical stadium `venue` is separate from `is_home`/`was_home` perspective.

If an observed WhoScored date disagrees with Understat, the builder uses the
Understat date only when a unique FPL fixture for the same home/away pair in
that season is finished and its kickoff date agrees with Understat. Review
`_date_reconciliation_audit.csv` for the original WhoScored date, corroborating
dates, fixture IDs, and selected played date. The audit is replaced on each
successful build (with headers only if no dates were reconciled). Provider
source files remain unchanged; conflicts without this agreement still fail.

## 13. Assign canonical match IDs to cleaned FPL gameweeks

Run when merged gameweek data exists; skip in a roster-only preseason build.
FBref summary data is an optional fallback, not a prerequisite.

```powershell
python -m fpl_assistant.providers.fpl.clean.assign_game_ids `
  --proc-root "data/processed/fpl/ENG-Premier League" `
  --fixture-calendar-root "data/processed/registry/fixtures" `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --tz UTC `
  --log-level INFO
```

This attaches canonical `match_id` and the legacy `game_id` alias. Official
FPL fixture IDs and gameweek numbers remain separate fields.

## 14. Scrape ClubElo and publish fixture ratings

Use today's date for active-season team discovery. The scraper's `--season`
option resolves a January snapshot in the season's end year, which can be in
the future during an autumn run. For a historical build, replace the date
below with an appropriate date within that historical season.

```powershell
$clubEloDate = Get-Date -Format "yyyy-MM-dd"
python -m fpl_assistant.providers.clubelo.scrape.clubelo_scraper `
  --league ENG_1 `
  --date $clubEloDate `
  --out-dir "data/raw/clubelo" `
  --max-age-days 0 `
  --verbose
```

Then clean histories and publish the provider-owned Elo schedule:

```powershell
python -m fpl_assistant.providers.clubelo.clean.clubelo_understat_enricher `
  --league "ENG-Premier League" `
  --season "2026-2027" `
  --fixture-root "data/processed/registry/fixtures"
```

Output includes `data/processed/clubelo/ENG-Premier League/2026-2027/schedule.csv`
with pre-match and season-frozen preseason Elo. Check coverage for all season
teams, especially promoted teams; historical builds may require team histories
for clubs no longer in today's league. Keep the default coverage guard.

## 15. Build team form

```powershell
python -m fpl_assistant.pipelines.integrate.team_form_builder `
  --season "2026-2027" `
  --fixtures-root "data/processed/registry/fixtures" `
  --out-dir "data/processed/registry/features" `
  --write-latest `
  --strict-ids `
  --force `
  --log-level INFO
```

If a materialized fixture/FDR view is wanted, rerun the stage 12 command now
with `--attach-fdr latest`. This is optional; it requires the team-form output.

## 16. Build the provider-observed player-fixture calendar

```powershell
python -m fpl_assistant.pipelines.integrate.calendar_builder `
  --fixtures-root "data/processed/registry/fixtures" `
  --whoscored-root "data/processed/whoscored/ENG-Premier League" `
  --fpl-root "data/processed/fpl/ENG-Premier League" `
  --features-root "data/processed/registry/features" `
  --team-version latest `
  --season "2026-2027" `
  --force `
  --log-level INFO
```

This publishes `player_fixture_calendar_observed.csv` under the season's
fixture registry. Do not use `--create-empty` to conceal missing production
inputs. The next stage owns the expanded modelling calendar.

## 17. Publish eligibility, availability, and complete DNP calendars

```powershell
python -m fpl_assistant.providers.fpl.pipelines.eligibility_backfill `
  --processed-fpl-root "data/processed/fpl/ENG-Premier League" `
  --raw-fpl-root "data/raw/fpl/ENG-Premier League" `
  --fixtures-root "data/processed/registry/fixtures" `
  --registry-root "data/processed/registry" `
  --understat-root "data/processed/understat/ENG-Premier League" `
  --season "2026-2027" `
  --force `
  --log-level INFO
```

This writes `player_fixture_calendar.csv`, effective-dated eligibility,
availability history, fixture snapshots, and reconstruction audits. Historical
zero-minute observations become explicit DNP rows. Future eligible fixtures
have pending rows with null minutes, not recorded DNPs.

Run this after stage 16 and before player form: player form consumes the
expanded calendar, not the provider-observed file. Understat player-match xG/xA
is joined here. Team context uses `team_gf`, `team_ga`, `team_xg`, `team_xga`;
player outcomes use `goals`, `assists`, `xg`, `xa`.

## 18. Build player form

```powershell
python -m fpl_assistant.pipelines.integrate.player_form_builder `
  --season "2026-2027" `
  --fixtures-root "data/processed/registry/fixtures" `
  --out-dir "data/processed/registry/features" `
  --write-latest `
  --force `
  --log-level INFO
```

Routine team/player form publication writes to
`data/processed/registry/features/latest/2026-2027/`. Use `--auto-version` only
when intentionally creating an immutable version snapshot.

## 19. Run final assurance

```powershell
python -m fpl_assistant.qa.assurance `
  --fixtures-root "data/processed/registry/fixtures" `
  --teams-lookup "data/processed/registry/_id_lookup_teams.json" `
  --players-lookup "data/processed/registry/_id_lookup_players.json" `
  --seasons "2026-2027" `
  --log-level INFO

python -m pytest -q
```

Assurance requires the fixture and player calendars. The optional FBref lineup
cross-check is skipped when that source is absent. Investigate failures rather
than lowering coverage thresholds. The full pytest suite checks repository
behavior; it does not prove that a live provider scrape was complete.

Before using the publication, confirm:

- Raw and processed FPL paths contain the league folder exactly once.
- The FPL roster has unique, non-null canonical player IDs plus team IDs,
  official positions, and FPL element IDs; generated registrations and bridges agree.
- Fixture dates belong to the target season and canonical keys are unique.
- Coverage audits distinguish unplayed fixtures from failed completed-match downloads.
- WhoScored/Understat identities resolve and FPL metric-enrichment audits preserve row counts.
- FPL gameweek match IDs agree with the canonical fixture calendar.
- Both observed and expanded player calendars exist; eligibility/DNP audits are reviewed.
- ClubElo ratings cover the season teams and both form outputs exist under `features/latest`.
- No deprecated FBref partial tables remain in the active processed tree.

## Preseason and repeat runs

For a season with no played fixtures, run FPL scraping, roster publication,
fixture bootstrap, WhoScored schedule discovery/partial cleaning, and ClubElo
publication. Prices and the FPL master can also publish; eligibility backfill
can construct pending roster-by-fixture rows from the bootstrap calendar.
Skip completed-event scraping, absent gameweek processing, and match-dependent
form stages until their inputs exist. Do not label that preseason subset a
complete played-match feature build. Carry-over performance is reset only when
the cleaner detects an entirely unstarted season with stale cumulative values;
do not assume a particular season remains preseason indefinitely.

For each active-season refresh, follow the same order. Refresh mutable provider
snapshots; reuse valid caches for interrupted downloads. Use `--skip-existing`
only when intentionally retaining existing outputs: it can preserve stale
aggregate files as well as successful downloads. Any provider rebuild should
be followed by the FPL metric refresh and downstream calendar/feature stages.
If only downstream processing failed, resume at the failed stage using the
validated upstream outputs rather than scraping everything again.

Detailed data contracts remain in [FPL_PIPELINE.md](FPL_PIPELINE.md),
[FBREF_PIPELINE.md](FBREF_PIPELINE.md), and
[FPL_ELIGIBILITY_BACKFILL.md](FPL_ELIGIBILITY_BACKFILL.md). Use this document
for the combined execution order.
