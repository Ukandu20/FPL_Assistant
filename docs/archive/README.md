# Historical documentation and artifacts

These files are preserved for reference. Their commands, paths, numerical
results, and assumptions are not the current operational contract. Use the
[documentation index](../README.md) and
[complete pipeline](../COMPLETE_DATA_PIPELINE.md) for maintained guidance.

| File | Why retained | Current reference |
|---|---|---|
| [commit-msg.txt](commit-msg.txt) | Old pipeline commit draft | Complete pipeline and migration report |
| [fbref_pipeline.txt](fbref_pipeline.txt) | Retired `scripts.*` commands and FBref-first assumptions | Complete pipeline; provider integration reference |
| [fpl_pipeline.txt](fpl_pipeline.txt) | Retired commands and unscoped FPL storage paths | Complete pipeline; FPL reference |
| [minutes.txt](minutes.txt) | Historical V1 model manual and appearance-only data assumptions | Expected-minutes V2 operations and explicit V1 compatibility workflow |
| [team_form_builder.txt](team_form_builder.txt) | Retired module paths, old defaults, and versioning guidance | Complete pipeline, team-form stage |
| [optimizer_single_gw_prototype.txt](optimizer_single_gw_prototype.txt) | Python prototype retained by architecture migration | Maintained `src/fpl_assistant/optimizers/` package |

`expected_points.csv` is a local historical export, not a documentation example
or production prediction source. It contains 16,188 records for 2024-2025 and
237 repeated `(player_id, season, gw_orig)` keys. Those repeats require fixture
or run provenance before they can be called erroneous duplicates. The file is
preserved locally and ignored by Git; a fresh checkout need not contain it.
No generating command, model version, or trustworthy cutoff is established here.

The six files moved into this folder during the documentation review were
preserved without modifying their contents. The optimizer prototype was
already archived. Do not copy historical commands into active runbooks without
checking them against current source.
