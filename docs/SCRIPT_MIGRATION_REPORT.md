# `scripts/` to `src/fpl_assistant/` migration report

Date: 2026-08-14

Status: audit and proposed migration plan only. No file in `scripts/` has been moved or deleted as part of this report.

## Executive summary

The repository contains 105 Python files under `scripts/`.

| Classification | Files | Proposed treatment |
|---|---:|---|
| Byte-for-byte duplicate with an existing packaged destination | 75 | Keep the `src/fpl_assistant/` copy; update references; then delete the `scripts/` copy |
| Duplicate except for line endings | 1 | Keep the `src/fpl_assistant/` copy; normalize through the normal formatter; then delete the `scripts/` copy |
| Exact duplicate with a non-name-matching packaged destination (`scripts/app/index.py`) | 1 | Keep `src/fpl_assistant/apps/control_panel.py`; update the app launcher; then delete the script copy |
| Full implementation exists only in `scripts/` | 1 | Migrate the implementation into `src/fpl_assistant/`, replacing the current reverse wrapper |
| `scripts/` compatibility wrapper pointing to `src` | 1 | Delete only after callers use the package path |
| Functional implementations have diverged | 16 | Reconcile into the package destination while preserving every newer change and update from both copies; never replace newer behavior with an older implementation |
| Package `__init__.py` metadata differs | 6 | Keep the package initializer; remove the legacy initializer with its directory |
| Script-only test/evaluation files | 4 | Relocate rather than discard; see the test disposition section |
| **Total** | **105** | |

The migration is not ready for deletion yet. There are 30 active `from scripts...` imports inside `src`, 16 more in the main `tests/` tree, and executable subprocess/module references to `scripts.*` in the control panel and pipeline orchestrators.

## Destination rules

The proposed canonical mapping is:

| Legacy area | Canonical destination |
|---|---|
| `scripts/clubelo_pipeline/` | `src/fpl_assistant/providers/clubelo/` |
| `scripts/fbref_pipeline/` | `src/fpl_assistant/providers/fbref/` |
| `scripts/fotmob_pipeline/` | `src/fpl_assistant/providers/fotmob/` |
| `scripts/fpl_pipeline/` | `src/fpl_assistant/providers/fpl/` |
| `scripts/transfermarkt_pipeline/` | `src/fpl_assistant/providers/transfermarkt/` |
| `scripts/understat_pipeline/` | `src/fpl_assistant/providers/understat/` |
| `scripts/whoscored_pipeline/` | `src/fpl_assistant/providers/whoscored/` |
| `scripts/models/` | `src/fpl_assistant/models/` |
| `scripts/optimizers/` | `src/fpl_assistant/optimizers/` |
| `scripts/pipelines/` | `src/fpl_assistant/pipelines/` |
| `scripts/infer/` | `src/fpl_assistant/infer/` |
| `scripts/qa/` | `src/fpl_assistant/qa/` |
| `scripts/tools/` | `src/fpl_assistant/tools/` |
| `scripts/utils/` | `src/fpl_assistant/utils/` |
| `scripts/cleaners/` | `src/fpl_assistant/cleaners/` |
| `scripts/common/cli.py` | `src/fpl_assistant/platform/cli.py` |
| `scripts/app/index.py` | `src/fpl_assistant/apps/control_panel.py` |

## Files requiring migration or merge

### Migrate the script implementation into `src`

| Source | Destination | Finding | Proposed operation |
|---|---|---|---|
| `scripts/clubelo_pipeline/clean/clubelo_understat_enricher.py` | `src/fpl_assistant/providers/clubelo/clean/clubelo_understat_enricher.py` | The source is the full implementation (about 1,025 lines); the destination is a three-line reverse wrapper importing `scripts.*` | Move the implementation to the destination, convert imports to package imports, test it, then delete the source |

This is the one file that must not be deleted on the assumption that the package copy already contains the implementation.

### Keep the packaged implementation and remove the legacy wrapper

| Legacy wrapper | Packaged implementation | Proposed operation |
|---|---|---|
| `scripts/whoscored_pipeline/clean/whoscored_cleaner.py` | `src/fpl_assistant/providers/whoscored/clean/whoscored_cleaner.py` | Update callers to the package module, test the package command, then delete the wrapper |

### Merge/reconcile functional differences

For these files, `src/fpl_assistant/` is the required destination. Reconciliation must preserve **all of the latest changes and updates** made in either copy. The current `src` differences contain the newer canonical-ID, league-scoped-path, provider-capability, position-schema, and package-import work, so those updates are mandatory and must not be rolled back.

The merge policy for every file in this section is:

1. Start with the current packaged implementation at the `src/fpl_assistant/` destination.
2. Compare the complete Git history and the current diff for both copies, not only file modification timestamps.
3. Retain every newer functional change already present in `src`.
4. Port any later or still-relevant change that exists only in `scripts/` into the packaged destination.
5. Resolve conflicts in favor of the newest intended behavior while retaining compatible improvements from both sides.
6. Do not restore obsolete paths, imports, schemas, or entry-point delegation merely because they remain in the legacy copy.
7. Add or update regression tests for each reconciled behavior before the legacy source is eligible for deletion.

Therefore, “keep `src`” in the table below means “keep the packaged file as the destination and preserve its latest updates, then merge any newer non-obsolete legacy-only update into it.” It does not mean blindly discarding unique changes from `scripts/`, and it does not permit overwriting the packaged file with an older script copy.

| Legacy source | Packaged destination | Important difference | Proposed result |
|---|---|---|---|
| `fbref_pipeline/clean/csv_cleaner.py` | `providers/fbref/clean/csv_cleaner.py` | Packaged version preserves canonical match identity and includes formatting cleanup | Preserve the latest packaged identity work; port any newer compatible script-only changes; verify identity-related tests |
| `fbref_pipeline/integrate/calendar_builder.py` | `providers/fbref/integrate/calendar_builder.py` | Packaged version switches player-match inputs toward WhoScored and changes metric/position handling | Preserve the latest provider and schema changes; merge any newer compatible legacy behavior; validate calendar and player-form outputs |
| `fbref_pipeline/integrate/fixtures_meta_builder.py` | `providers/fbref/integrate/fixtures_meta_builder.py` | Packaged version adds preseason fixture-calendar bootstrap and stable canonical IDs | Preserve bootstrap and canonical-ID updates; merge any newer compatible script-only change; run fixture bootstrap tests |
| `fbref_pipeline/integrate/player_form_builder.py` | `providers/fbref/integrate/player_form_builder.py` | Packaged version standardizes goalkeeper position from `GK` to `GKP` | Preserve the latest `GKP` schema; merge newer compatible calculations without restoring the old schema; validate downstream schemas |
| `fbref_pipeline/scrape/fbref_adapter.py` | `providers/fbref/scrape/fbref_adapter.py` | Package imports and stable capability aliases | Preserve the latest package imports and capability aliases; port any newer compatible adapter changes |
| `fbref_pipeline/scrape/match_stats_scraper.py` | `providers/fbref/scrape/match_stats_scraper.py` | Relative package imports and first-class lineup/event capabilities | Preserve the latest capability and package-import work; merge newer compatible scraper behavior; update callers and tests |
| `fbref_pipeline/scrape/season_stats_scraper.py` | `providers/fbref/scrape/season_stats_scraper.py` | Relative package imports | Preserve the latest package imports; port any newer compatible scraping changes |
| `fbref_pipeline/scrape/whoscored_match_stats_scraper.py` | `providers/fbref/scrape/whoscored_match_stats_scraper.py` | Packaged version uses central platform paths | Preserve central path handling; merge any newer compatible scraper updates |
| `fbref_pipeline/utils/fbref_utils.py` | `providers/fbref/utils/fbref_utils.py` | Correct installed-package and relative imports | Preserve installed-package behavior; port any newer compatible utility updates |
| `fpl_pipeline/clean/assign_game_ids.py` | `providers/fpl/clean/assign_game_ids.py` | Packaged version replaces stale IDs with canonical FBref IDs and writes bridge records | Preserve all canonical-ID and bridge updates; merge any newer compatible assignment logic; run canonical bridge tests |
| `fpl_pipeline/clean/gw_stats_cleaner.py` | `providers/fpl/clean/gw_stats_cleaner.py` | Packaged version adds league scoping, official element mapping, and `GKP` normalization | Preserve all latest league, identity, and position updates; port newer compatible cleaning logic; run FPL cleaner tests |
| `fpl_pipeline/master/consolidate_master.py` | `providers/fpl/master/consolidate_master.py` | Packaged version uses league-scoped roots and owns its `main()` | Preserve the latest package CLI and league paths; port any newer compatible consolidation updates; replace old commands |
| `fpl_pipeline/pipelines/clean_and_enrich.py` | `providers/fpl/pipelines/clean_and_enrich.py` | Packaged version is substantially newer, with cumulative-stat, preseason, identity, and league-scoping work | Preserve every latest packaged pipeline update; merge any newer non-obsolete legacy-only behavior; regression-test the full pipeline |
| `fpl_pipeline/pipelines/prices_from_merged.py` | `providers/fpl/pipelines/prices_from_merged.py` | Packaged CLI and league-scoped paths | Preserve the latest package CLI and league paths; port any newer compatible pricing changes |
| `fpl_pipeline/scrape/season_scraper.py` | `providers/fpl/scrape/season_scraper.py` | Packaged version uses league-scoped raw paths | Preserve the latest league-scoped behavior; port newer compatible scraper updates; replace remaining `scripts.*` imports |
| `understat_pipeline/clean/clean_understat_raw.py` | `providers/understat/clean/clean_understat_raw.py` | Packaged version adds official FPL aliases and canonical bridge integration | Preserve all latest alias and bridge updates; merge newer compatible cleaning behavior; run Understat and bridge tests |

### Package initializer differences

These are not competing implementations. The packaged initializers contain package exports or updated descriptions and should be retained:

- `clubelo_pipeline/clean/__init__.py`
- `fpl_pipeline/__init__.py`
- `models/__init__.py`
- `optimizers/__init__.py`
- `transfermarkt_pipeline/__init__.py`
- `whoscored_pipeline/clean/__init__.py`

The corresponding `scripts/` initializers can be deleted with their legacy directories after imports have been migrated.

## Direct duplicate deletion candidates

These files already have byte-identical packaged copies. They are deletion candidates only after all imports, subprocess module names, documentation, and tests point to the packaged locations.

### ClubElo

- `enrich/add_fbref_pl_matches.py`
- `enrich/add_pl_season_bands.py`
- `enrich/add_transfermarkt_managers.py`
- `scrape/clubelo_scraper.py`

### FBref

- `automation/auto_scrape.py`
- `automation/fbref_automated_scrape.py`
- `clean/new.py`
- `clean/world_cup_cleaner.py`
- `scrape/__init__.py`
- `scrape/fbref_robust.py`
- `scrape/roster_fetcher.py`
- `scrape/whoscored_match_stats_scraper_soccerdata.py`
- `scrape/whoscored_native_backend.py`
- `scrape/whoscored_scraper.py`
- `utils/scrape_meta.py`

`integrate/team_form_builder.py` is also functionally identical; its only current difference is line endings.

### FPL provider

- `analysis/__init__.py`
- `analysis/aggregated_points_goals.py`
- `analysis/gw_data_collector.py`
- `clean/__init__.py`
- `clean/cleaners.py`
- `master/__init__.py`
- `pipelines/__init__.py`
- `scrape/__init__.py`
- `scrape/api_client.py`
- `scrape/cron_generator.py`
- `scrape/gameweek.py`
- `scrape/teams_scraper.py`
- `scrape/top_managers.py`
- `scrape/top_players.py`
- `utils/__init__.py`
- `utils/file_utils.py`
- `utils/global_merger.py`
- `utils/mergers.py`
- `utils/parse_helpers.py`
- `utils/position_checker.py`
- `utils/utility.py`

### Models and inference

- `infer/predict_upcoming_minutes.py`
- `models/captain_ranker.py`
- `models/defense_forecast.py`
- `models/defense_model_builder.py`
- `models/discipline_model_builder.py`
- `models/expected_points_aggregator.py`
- `models/goals_assists_forecast.py`
- `models/goals_assists_model_builder.py`
- `models/minutes_forecast.py`
- `models/minutes_model_builder.py`
- `models/points_forecast.py`
- `models/predicted_minutes.py`
- `models/saves_forecast.py`
- `models/saves_model_builder.py`
- `models/squad_optimizer.py`
- `models/three_gw_optimizer.py`
- `models/total_points_combiner.py`

### Optimizers and orchestration

- `optimizers/availability.py`
- `optimizers/build_optimizer.py`
- `optimizers/mc.py`
- `optimizers/multi_gw.py`
- `optimizers/multi_gw_hold.py`
- `optimizers/simulator.py`
- `optimizers/single_gw.py`
- `optimizers/strategy.py`
- `optimizers/team_state.py`
- `pipelines/forecaster.py`
- `pipelines/model_builder.py`

### Other providers and utilities

- `cleaners/clean_players.py`
- `common/cli.py`
- `fotmob_pipeline/scrape/fotmob_stats_scraper.py`
- `qa/__init__.py`
- `qa/assurance.py`
- `tools/apply_transfers.py`
- `transfermarkt_pipeline/scrape/__init__.py`
- `transfermarkt_pipeline/scrape/manager_history_scraper.py`
- `understat_pipeline/clean/propose_player_aliases.py`
- `understat_pipeline/scrape/understat_stats_scraper.py`
- `utils/validate.py`

### App

- `scripts/app/index.py` is byte-identical to `src/fpl_assistant/apps/control_panel.py`, despite the different filename. Keep the packaged control panel and update the launch command before deleting `scripts/app/index.py`.

## Script-only test and evaluation files

These four files do not have current package counterparts and must not be deleted without relocation:

| File | Type | Recommended destination |
|---|---|---|
| `scripts/tests/backtest_harness.py` | Executable evaluation harness | `src/fpl_assistant/testing/backtest_harness.py`, plus a package CLI entry if it is still used |
| `scripts/tests/data_inputs.py` | Pytest data-quality suite | `tests/test_data_inputs.py` |
| `scripts/tests/optimizer_invariants.py` | Pytest optimizer-invariant suite | `tests/test_optimizer_invariants.py` |
| `scripts/tests/regression_golden.py` | Pytest golden regression suite | `tests/test_regression_golden.py` |

The three pytest files should remain in the repository test tree rather than becoming production package modules. The backtest harness is executable application logic and fits the existing `fpl_assistant.testing` package.

## References that block deletion

### Active imports

- 30 active `from scripts...` or `import scripts...` lines remain under `src/`.
- 16 active legacy imports remain under `tests/`.
- Important affected areas include model validation imports, FPL utilities and scrapers, FBref automation, WhoScored adapters, and the ClubElo/FotMob/Understat scrape metadata hook.

All `src` imports must be rewritten to `fpl_assistant.*` or safe relative package imports. Tests must then import the packaged modules so that the tests exercise the code that will actually ship.

### Executable module names

The following runtime orchestration still launches legacy modules:

- `src/fpl_assistant/apps/control_panel.py`: five model forecast commands and two optimizer commands.
- `src/fpl_assistant/pipelines/forecaster.py`: five model forecast commands.
- `src/fpl_assistant/pipelines/model_builder.py`: four model-builder commands.
- `src/fpl_assistant/providers/fbref/automation/fbref_automated_scrape.py`: two dynamically selected scraper module names.

These strings must be changed to `fpl_assistant.*` before `scripts/` is removed.

### Documentation and examples

Legacy paths also remain in docstrings and documentation, including `docs/FPL_PIPELINE.md`, `docs/MODEL_EVALUATION_REPORT.md`, and command examples inside several packaged model, optimizer, and tool modules. These do not all break imports, but leaving them unchanged would direct users back to deleted commands.

## Proposed migration sequence

1. Move the full ClubElo–Understat enricher implementation into its package destination.
2. Make `src/fpl_assistant/` self-contained by replacing all 30 active imports from `scripts.*`.
3. Replace subprocess and dynamic module names with `fpl_assistant.*` module paths.
4. Reconcile the 16 divergent implementations using the recency-preserving merge policy above: retain every latest packaged update, port every newer non-obsolete script-only change, and verify each preserved behavior with tests.
5. Move the four script-only test/evaluation files to their recommended destinations.
6. Update the 16 legacy imports in `tests/` and update documentation/command examples.
7. Run focused provider, canonical-ID, FPL-cleaning, model, optimizer, and app tests.
8. Run the complete test suite from an environment where `scripts/` is temporarily unavailable or renamed. This is the strongest check that no hidden dependency remains.
9. Present the resulting diff and test evidence for approval.
10. Only after approval, delete the obsolete `scripts/` files/directories in a separate, clearly scoped change.

## Deletion gate

No `scripts/` file should be deleted until all of the following are true:

- `rg` finds no active Python import of `scripts.*` outside the legacy tree.
- No subprocess or dynamic module name launches `scripts.*`.
- Tests import `fpl_assistant.*` and pass.
- The ClubElo–Understat implementation exists in `src`, not behind a reverse wrapper.
- Every divergent pair has a documented Git-history/diff review showing that all newer changes from both copies were preserved in `src`.
- The four script-only test/evaluation files have approved destinations.
- Documentation points to packaged commands.
- A full test run passes with the legacy tree unavailable.
