# Documentation review

Review date: 2026-09-17. Scope: all 26 files present under `docs/`, including
ignored files and the existing archive. This is a documentation, command, and
repository-reference review, not a fresh statistical evaluation of the models
or an external verification of provider methodology.

## Findings and cleanup

The former `docs/*` ignore rule exposed only eight files to `rg --files`.
Eighteen paths matched ignore rules, including one already tracked operations
document; nine files were tracked in total. Seventeen files had no tracked
copy, including the combined pipeline, minutes contracts, eligibility
documentation, and archetype catalogue.
The blanket rule has been removed; local archived CSV exports remain ignored.

Five obsolete top-level text files competed with maintained commands. They and
the historical prediction CSV have moved to `archive/`, with their contents
preserved. No documentation or data was deleted.

The combined pipeline is now explicitly the owner of execution order; FPL and
provider documents are references. Minutes specification precedence is explicit,
historical reviews have dated status notes, and the signal-profile draft now
acknowledges existing implementation without approving unresolved decisions.
A PowerShell-incompatible angle-bracket timestamp placeholder was replaced by
a quoted example. Eligibility documentation now records null pending outcomes.

## File-by-file assessment

Paths below are relative to this folder and refer to the post-cleanup location.

| Original file | Assessment | Disposition |
|---|---|---|
| `COMPLETE_DATA_PIPELINE.md` | Current combined runbook; previously ignored | Keep as primary execution guide; enable tracking |
| `FPL_PIPELINE.md` | Useful contracts, but duplicated ordering looked authoritative | Keep as reference; clarify interleaving with providers |
| `FBREF_PIPELINE.md` | Covers multiple providers despite historical filename | Keep stable filename; label as provider reference |
| `FPL_ELIGIBILITY_BACKFILL.md` | Current calendar ownership and leakage contract; preseason phrasing incomplete | Retain; document active-season pending rows and null outcomes |
| `FPL_EXPECTED_MINUTES_V2.md` | Current CLI operations; dates and paths are examples | Retain; link governing contract and clarify example inputs |
| `FPL_expected_minutes_V2_implementation_contract_HARDENED.md` | Explicitly locked V2.0 contract | Retain as governing specification; link operations |
| `FPL_expected_minutes_V2_implementation_decision_contract_completed.md` | Earlier defaults compete with later hardening | Mark superseded; preserve decision history in place |
| `FPL_expected_minutes_model_v2_design.md` | Initial proposal with stale local-source and latest-run context | Mark historical/superseded; preserve reasoning |
| `FPL_ARCHETYPE_V1.md` | Current implementation/publication reference; historical validation results | Retain; fix PowerShell timestamp example and label example cutoff |
| `fpl_archetype_scheme_catalogue.md` | Governing detailed catalogue, not redundant operational prose | Retain intact; enable tracking and index as authoritative |
| `FPL_player_archetype_implementation_decision_template.md` | Populated workbook despite template name; unresolved sign-off | Retain; explicitly subordinate to catalogue, no inferred approval |
| `FPL_ARCHETYPE_V1_SAMPLE_OUTPUT.json` | Valid JSON with four synthetic position examples | Retain unchanged; index as synthetic, not forecast data |
| `FPL_APP_DESIGN.md` | Product navigation and evidence rules with existing app entry point | Retain unchanged as product contract |
| `FPL_PLAYER_SIGNAL_PROFILE_DESIGN.md` | Draft includes unresolved choices while signal UI already exists | Add implementation/status note; preserve open decisions |
| `PLAYER_PROFILES.md` | Valid legacy season-profile command; easily confused with archetypes | Retain; distinguish artifact families and example season |
| `CLUBELO_FALLBACK_DESIGN.md` | Isolated fallback contract with existing implementation/tests | Retain; distinguish from normal acquisition and conceptual methodology |
| `clubelo_methodology.md` | General external-methodology notes lack verification provenance | Retain as conceptual reference; not an implementation guarantee |
| `MODEL_EVALUATION_REPORT.md` | Dated evaluation with historical artifact references | Retain in place; explicitly not a current evaluation |
| `SCRIPT_MIGRATION_REPORT.md` | Useful old-to-new mappings and completed-run verification counts | Retain in place with historical status; do not rewrite old paths |
| `commit-msg.txt` | Old commit prose describes retired pipeline assumptions | Archive as `archive/commit-msg.txt` |
| `fbref_pipeline.txt` | Duplicates modern runbook using retired `scripts.*` commands | Archive as `archive/fbref_pipeline.txt` |
| `fpl_pipeline.txt` | Retired commands, unscoped paths, outdated prerequisite text | Archive as `archive/fpl_pipeline.txt` |
| `minutes.txt` | V1-era guide assumes appearance-only calendars | Archive as `archive/minutes.txt`; V1 compatibility remains documented in V2 operations |
| `team_form_builder.txt` | Old CLI/path/default notes conflict with current routine publication | Archive as `archive/team_form_builder.txt` |
| `expected_points.csv` | 2024-2025 generated output, not documentation; incomplete provenance | Preserve as local ignored `archive/expected_points.csv` |
| `archive/optimizer_single_gw_prototype.txt` | Already historical Python prototype | Keep untouched in archive; do not present as production entry point |

## Validation and limits

Review checks cover local Markdown links, documented package module paths,
argument names against source declarations, PowerShell syntax, JSON syntax,
CSV shape, and Git visibility. Historical commands inside archived text and
dated migration/evaluation reports remain historical evidence, not runnable
instructions. Source inspection does not prove live scrape success or validate
all CLI argument combinations and external prerequisites.

Results: 78 PowerShell blocks parsed, 75 local Markdown links resolved, and
70 active package/console command examples matched existing modules and argument
names. The synthetic JSON contains four valid records. Archived CSV metadata
was inspected without modifying its records. No scrapers, training jobs, or
production data pipelines were run for this review.

Large design contracts were assessed for status, overlap, references, and
implementation boundaries. Their formulas, parameter choices, statistical
claims, and owner decisions were not independently revalidated or rewritten.
External methodology and historical model metrics require a separate evidence
review before being promoted to fresh production claims.

Some active reference documents intentionally repeat focused commands from the
combined runbook. They remain because their surrounding schema, troubleshooting,
and audit context is useful; precedence is now explicit. Superseded design
filenames remain stable to avoid breaking external references.
