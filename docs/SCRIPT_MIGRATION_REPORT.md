# Architecture consolidation report

Date: 2026-09-10

## Final directory ownership

- `src/fpl_assistant/providers/<provider>/`: provider-specific scraping, cleaning,
  enrichment, and FPL season publication.
- `src/fpl_assistant/pipelines/integrate/`: canonical fixture calendars and
  combined player/team feature builders, formerly housed under FBref.
- `src/fpl_assistant/platform/scrape_runs.py`: shared scrape job identity,
  run metadata, and scheduling. Reads structured run records and historical
  timestamp-only records without rewriting existing data.
- `src/fpl_assistant/domain/team_state.py`: the single team-state implementation.
- `models/`, `minutes_v2/`, `archetypes/`, `optimizers/`, `qa/`, `apps/`, and
  `testing/` remain separate responsibilities within the application package.
- `tests/`: regression tests plus the relocated optimizer/data/golden checks.
- `scripts/`: only the archetype sample generator; no production implementations.

The cleanup preserves the worktree's existing data, model behavior, canonical
identity changes, and in-progress user edits. It does not regenerate datasets,
train models, launch scrapers, or publish new artifacts into production folders.

## Commands that moved

| Old module or file | Maintained location |
|---|---|
| `scripts.<provider>_pipeline.*` | `fpl_assistant.providers.<provider>.*`, subject to the moves below |
| `fpl_assistant.providers.fbref.scrape.whoscored_*` | `fpl_assistant.providers.whoscored.scrape.whoscored_*` |
| `fpl_assistant.providers.fbref.integrate.*` | `fpl_assistant.pipelines.integrate.*` |
| `scripts.models.*` | `fpl_assistant.models.*` |
| `scripts.models.minutes_v2` | `fpl_assistant.minutes_v2.cli` |
| `scripts.optimizers.team_state` / `fpl_assistant.optimizers.team_state` | `fpl_assistant.domain.team_state` |
| `scripts.optimizers.*` | `fpl_assistant.optimizers.*` |
| `scripts.pipelines.*` | `fpl_assistant.pipelines.*` |
| `scripts.app.index` | `fpl_assistant.apps.control_panel` |
| `scripts.tests.backtest_harness` | `fpl_assistant.testing.backtest_harness` |
| `scripts/tests/{data_inputs,optimizer_invariants,regression_golden}.py` | `tests/test_{data_inputs,optimizer_invariants,regression_golden}.py` |

Use `python -m <module>` for CLI modules. The Streamlit control panel launches
with `streamlit run src/fpl_assistant/apps/control_panel.py`. External scheduled
jobs using removed paths must adopt the mappings above. Repository call sites,
runbooks, and notebook source references are updated.

WhoScored's main entry point retains native and soccerdata backend selection.
Its schedule-only CLI is a small argument adapter; its soccerdata adapter now
forwards shell arguments correctly. The unused duplicate schedule scraping body
was removed. Native JSON extraction, blocked-page handling, and cached season
recovery from the newer package implementation are retained.

## Reconciliation evidence

The initial inventory contained 107 Python files under `scripts/`: 73 byte-identical
package counterparts, 29 different counterparts (including wrappers and metadata),
and five unique files. Exact duplicates retain the packaged implementation.
The sample generator stays in place; the other four unique files were relocated.

For divergent pairs, current source differences and Git history were inspected.
The package implementation was retained except for the ClubElo enricher, whose
full implementation was copied out of its reverse wrapper. No legacy-only
function/class definitions were found in the other divergent implementations.
Changes retained include FPL league scoping and official element mappings,
`events.csv`, `GKP` normalization, preseason and cumulative-stat enrichment,
canonical FBref match identity, Understat aliases/bridges, package imports,
and the newer WhoScored native backend. Minutes-builder differences were comments.

| Legacy source | Destination | Last legacy / package commits reviewed |
|---|---|---|
| `scripts/clubelo_pipeline/clean/__init__.py` | `src/fpl_assistant/providers/clubelo/clean/__init__.py` | dc6b620 2026-06-06 feat: add provider ingestion pipelines / 2fa6ddc 2026-08-07 feat: preserve canonical IDs in ClubElo enrichment |
| `scripts/clubelo_pipeline/clean/clubelo_understat_enricher.py` | `src/fpl_assistant/providers/clubelo/clean/clubelo_understat_enricher.py` | 075df8e 2026-08-14 feat(fixtures): bootstrap preseason provider schedules / 2fa6ddc 2026-08-07 feat: preserve canonical IDs in ClubElo enrichment |
| `scripts/fbref_pipeline/clean/csv_cleaner.py` | `src/fpl_assistant/providers/fbref/clean/csv_cleaner.py` | 20e4a6a 2026-08-07 feat: adapt FBref pipeline to reduced coverage / 20e4a6a 2026-08-07 feat: adapt FBref pipeline to reduced coverage |
| `scripts/fbref_pipeline/integrate/calendar_builder.py` | `src/fpl_assistant/pipelines/integrate/calendar_builder.py` | f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral / f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral |
| `scripts/fbref_pipeline/integrate/fixtures_meta_builder.py` | `src/fpl_assistant/pipelines/integrate/fixtures_meta_builder.py` | f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral / e02fc6c 2026-08-25 fix(fixtures): recognize completed current-season matches |
| `scripts/fbref_pipeline/integrate/player_form_builder.py` | `src/fpl_assistant/pipelines/integrate/player_form_builder.py` | f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral / f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral |
| `scripts/fbref_pipeline/integrate/team_form_builder.py` | `src/fpl_assistant/pipelines/integrate/team_form_builder.py` | f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral / f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral |
| `scripts/fbref_pipeline/scrape/fbref_adapter.py` | `src/fpl_assistant/providers/fbref/scrape/fbref_adapter.py` | 20e4a6a 2026-08-07 feat: adapt FBref pipeline to reduced coverage / 20e4a6a 2026-08-07 feat: adapt FBref pipeline to reduced coverage |
| `scripts/fbref_pipeline/scrape/match_stats_scraper.py` | `src/fpl_assistant/providers/fbref/scrape/match_stats_scraper.py` | 20e4a6a 2026-08-07 feat: adapt FBref pipeline to reduced coverage / 20e4a6a 2026-08-07 feat: adapt FBref pipeline to reduced coverage |
| `scripts/fbref_pipeline/scrape/season_stats_scraper.py` | `src/fpl_assistant/providers/fbref/scrape/season_stats_scraper.py` | 909a021 2026-07-30 feat: add canonical data architecture / 909a021 2026-07-30 feat: add canonical data architecture |
| `scripts/fbref_pipeline/scrape/whoscored_match_stats_scraper.py` | `src/fpl_assistant/providers/whoscored/scrape/whoscored_match_stats_scraper.py` | 1782662 2026-08-25 fix(whoscored): recover current-season match data / 1782662 2026-08-25 fix(whoscored): recover current-season match data |
| `scripts/fbref_pipeline/scrape/whoscored_native_backend.py` | `src/fpl_assistant/providers/whoscored/scrape/whoscored_native_backend.py` | f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral / 1782662 2026-08-25 fix(whoscored): recover current-season match data |
| `scripts/fbref_pipeline/utils/fbref_utils.py` | `src/fpl_assistant/providers/fbref/utils/fbref_utils.py` | 909a021 2026-07-30 feat: add canonical data architecture / 909a021 2026-07-30 feat: add canonical data architecture |
| `scripts/fpl_pipeline/__init__.py` | `src/fpl_assistant/providers/fpl/__init__.py` | 76de103 2025-08-02 updated directories and namings for scrapers and cleaners / dc6b620 2026-06-06 feat: add provider ingestion pipelines |
| `scripts/fpl_pipeline/clean/assign_game_ids.py` | `src/fpl_assistant/providers/fpl/clean/assign_game_ids.py` | 66bc72b 2026-08-21 feat(fpl): publish timestamp-safe roster eligibility / 66bc72b 2026-08-21 feat(fpl): publish timestamp-safe roster eligibility |
| `scripts/fpl_pipeline/clean/gw_stats_cleaner.py` | `src/fpl_assistant/providers/fpl/clean/gw_stats_cleaner.py` | 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline / 9d621ac 2026-08-25 fix(fpl): fill canonical teams and official player metrics |
| `scripts/fpl_pipeline/master/consolidate_master.py` | `src/fpl_assistant/providers/fpl/master/consolidate_master.py` | 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline / 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline |
| `scripts/fpl_pipeline/pipelines/clean_and_enrich.py` | `src/fpl_assistant/providers/fpl/pipelines/clean_and_enrich.py` | 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline / 9d621ac 2026-08-25 fix(fpl): fill canonical teams and official player metrics |
| `scripts/fpl_pipeline/pipelines/prices_from_merged.py` | `src/fpl_assistant/providers/fpl/pipelines/prices_from_merged.py` | 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline / 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline |
| `scripts/fpl_pipeline/scrape/season_scraper.py` | `src/fpl_assistant/providers/fpl/scrape/season_scraper.py` | 93cb804 2026-08-10 feat(fpl): harden league-scoped preseason pipeline / 020f0d1 2026-08-21 feat(fpl): enhance gameweek hub insights |
| `scripts/models/__init__.py` | `src/fpl_assistant/models/__init__.py` | 61b6766 2025-08-04 Refactor team form builder to support both defensive and attacking metrics; update schema to v1.3 and enhance rolling calculations. Add calendar builder script for generating player minutes calendar from fixture data and player rosters. Create init file for models package and implement predicted minutes model for training and evaluation of expected minutes for FPL players. / 2f826c4 2026-06-06 feat: add forecasting and optimizer workflows |
| `scripts/models/minutes_model_builder.py` | `src/fpl_assistant/models/minutes_model_builder.py` | b139a40 2026-08-21 feat(minutes): implement hardened expected-minutes v2 / b139a40 2026-08-21 feat(minutes): implement hardened expected-minutes v2 |
| `scripts/models/minutes_v2.py` | `src/fpl_assistant/minutes_v2/cli.py` | b139a40 2026-08-21 feat(minutes): implement hardened expected-minutes v2 / b139a40 2026-08-21 feat(minutes): implement hardened expected-minutes v2 |
| `scripts/optimizers/__init__.py` | `src/fpl_assistant/optimizers/__init__.py` | 44f7e49 2025-09-06 feat(team_state): end-to-end team_state CLI with master_fpl integration + idempotent seeding / 2f826c4 2026-06-06 feat: add forecasting and optimizer workflows |
| `scripts/qa/assurance.py` | `src/fpl_assistant/qa/assurance.py` | f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral / f06394b 2026-08-21 refactor(fbref): make fixture integration provider-neutral |
| `scripts/transfermarkt_pipeline/__init__.py` | `src/fpl_assistant/providers/transfermarkt/__init__.py` | 9bea48d 2026-04-06 feat: add transfermarkt manager history scraper / dc6b620 2026-06-06 feat: add provider ingestion pipelines |
| `scripts/understat_pipeline/clean/clean_understat_raw.py` | `src/fpl_assistant/providers/understat/clean/clean_understat_raw.py` | a197bc6 2026-08-07 feat: harden Understat scraping and normalization / a197bc6 2026-08-07 feat: harden Understat scraping and normalization |
| `scripts/whoscored_pipeline/clean/__init__.py` | `src/fpl_assistant/providers/whoscored/clean/__init__.py` | 64bfe23 2026-08-07 feat: add native WhoScored cleaning pipeline / 64bfe23 2026-08-07 feat: add native WhoScored cleaning pipeline |
| `scripts/whoscored_pipeline/clean/whoscored_cleaner.py` | `src/fpl_assistant/providers/whoscored/clean/whoscored_cleaner.py` | 64bfe23 2026-08-07 feat: add native WhoScored cleaning pipeline / 1782662 2026-08-25 fix(whoscored): recover current-season match data |

## Other duplicated or historical code

- Tracked bytecode belonging to the removed source locations was deleted.
- The identical team-state implementation under `optimizers/` was removed in
  favor of `domain/team_state.py`.
- The old FBref `clean/new.py` registry cleaner was superseded by `csv_cleaner.py`;
  all its function names already exist in the maintained cleaner.
- `src/extras/scrape_fpl.py` moved to `providers/fpl/scrape/archive_scraper.py`.
  Its distinct Wayback/historical JSON workflow is retained.
- Archived JSON loaders/export live under `providers/fpl/clean/archive_tables.py`
  and `archive_export.py`. Importing the exporter no longer writes files.
- The player/team rename-rule cleaners are consolidated into
  `providers/fbref/clean/table_cleaner.py`; use `--entity player` (default) or
  `--entity team`. Team mode retains the `pl` -> `no_of_players_used` rename.
- The two historical FBref bulk exporters are one
  `tools/legacy/fbref_bulk_export.py` implementation with `--players-only`.
- Distinct historical-format tools (numeric-ID registries, rule-based registry
  imports, override matching, FantasyNutmeg history) are named explicitly under
  `tools/legacy/`. They are not used by current provider pipelines. These have
  different schemas/behavior and are not interchangeable with canonical tools.
- Ancillary files found during deletion were preserved: the fixture notebook
  moved to `notebooks/game_id.ipynb`, the TOML profile to
  `config/profiles/2025-2026.toml`, and the optimizer draft text to
  `docs/archive/optimizer_single_gw_prototype.txt`.

## Verification

- Before cleanup: 309 tests passed.
- After initial package migration: 312 passed, three artifact-dependent checks
  skipped (their optional input artifacts were absent).
- Final verification after deleting legacy implementations: **324 passed, three
  skipped**, with no warnings.
  Command: `python -m pytest -q -p no:cacheprovider --basetemp artifacts/test_runs/architecture_cleanup_verified --tb=short`.
- All **74 argparse CLI entry points** returned help successfully, without
  scraping, training, or changing production data.
- An offline wheel build passed; its Python sources match the current package,
  with no stale modules or historical model/data directories included.
- All 107 original script inventory entries have a retained destination; 106
  legacy Python source files were removed (the sample generator remains).
- Architecture regression tests check retired imports/launch targets, mirrored
  implementations, and the absence of production code in `scripts/`.
- Scheduling tests cover current and historical metadata; adapter tests preserve
  WhoScored command-line arguments.

## Pre-existing limitation found by CLI verification

The historical strategy evaluator expects an external `mc_sim_v01` module with
`SimConfig` and `run_sim`. That engine is absent from the repository, archives,
and its tracked Git history. The packaged `optimizers/mc.py` exposes a different
simulation contract, so replacing the engine would change the model and was not
part of this structural cleanup. Model and optimizer package exports are lazy;
this missing optional engine no longer prevents independent optimizer commands
from running. The strategy evaluator explains the dependency when evaluation is
requested, and its help remains available. No claim is made that this historical
strategy workflow or live external scraping has been exercised end to end.

Historical trained artifacts already under `src/data/` are preserved. Package
discovery is explicitly restricted to `fpl_assistant*`, keeping these artifacts
out of the wheel. Current workflows use the unchanged configured data locations.
