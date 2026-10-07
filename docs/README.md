# Documentation index

Start with [Complete data pipeline](COMPLETE_DATA_PIPELINE.md) for the ordered
scraping, cleaning, processing, and assurance commands. All commands assume the
repository root unless stated otherwise. Example seasons and timestamps are
inputs to replace deliberately, not automatically current values.

## Operations and data contracts

| Document | Use |
|---|---|
| [Complete data pipeline](COMPLETE_DATA_PIPELINE.md) | Primary multi-provider execution order |
| [Archetyping and modelling](ARCHETYPING_AND_MODELLING.md) | Post-processing methodology, model commands, publication order, and integration limits |
| [FPL reference](FPL_PIPELINE.md) | Roster, identity, enrichment, price, and FPL publication details |
| [Provider integration reference](FBREF_PIPELINE.md) | Optional FBref, WhoScored troubleshooting, provider cleaning and integration |
| [Eligibility and DNP reconstruction](FPL_ELIGIBILITY_BACKFILL.md) | Observed versus expanded calendars and timestamp safety |
| [Expected-minutes V2 operations](FPL_EXPECTED_MINUTES_V2.md) | Audit, training, forecasting, shadow comparison, and acceptance |
| [Archetypes V1](FPL_ARCHETYPE_V1.md) | Versioned archetype publication and validation |
| [Player production profiles](PLAYER_PROFILES.md) | Legacy season-profile publication, distinct from V1 archetypes |

The combined runbook owns cross-provider execution order. The FPL/provider
references retain detailed contracts and focused examples, not alternative
end-to-end sequences. Data processing does not automatically train models or
approve forecasts for production.

## Specifications and design

| Document | Status / precedence |
|---|---|
| [Application design](FPL_APP_DESIGN.md) | Product navigation and evidence contract |
| [Archetype catalogue](fpl_archetype_scheme_catalogue.md) | Governing V1 model definitions |
| [Archetype decision workbook](FPL_player_archetype_implementation_decision_template.md) | Supporting choices; subordinate to catalogue despite legacy filename |
| [Archetype sample output](FPL_ARCHETYPE_V1_SAMPLE_OUTPUT.json) | Four synthetic example records, not predictions |
| [Hardened minutes V2 contract](FPL_expected_minutes_V2_implementation_contract_HARDENED.md) | Governing V2.0 specification |
| [Earlier minutes design](FPL_expected_minutes_model_v2_design.md) | Historical proposal; superseded by hardened contract |
| [Earlier completed minutes decisions](FPL_expected_minutes_V2_implementation_decision_contract_completed.md) | Superseded defaults snapshot; retained decision history |
| [Player Signal Profile design](FPL_PLAYER_SIGNAL_PROFILE_DESIGN.md) | Design record with unresolved decisions; some UI already implemented |
| [ClubElo fallback contract](CLUBELO_FALLBACK_DESIGN.md) | Isolated fallback implementation; explicit activation and coverage guards |
| [ClubElo methodology](clubelo_methodology.md) | Conceptual reference; broader than the local fallback implementation |

Keep governing specifications separate from their historical proposals.
Implementation or a design document alone does not establish model acceptance.

## Reviews and history

- [Documentation review](DOCUMENTATION_REVIEW.md): assessment and disposition of every original file.
- [Architecture migration](SCRIPT_MIGRATION_REPORT.md): completed migration and historical path mappings.
- [Model evaluation](MODEL_EVALUATION_REPORT.md): findings from 2026-08-09, not a current model certification.
- [Archive](archive/README.md): obsolete runbooks, V1 notes, prototype, and local data snapshot.

## Maintenance

Update the combined runbook when stage ordering or package entry points change.
Update the owning reference when an output schema or invariant changes. Label
historical reports and superseded designs instead of silently rewriting their
results. Check PowerShell syntax and source argument declarations after command
edits; do not run live scrapes just to validate documentation.

Markdown, JSON examples, and archived text belong in version control. Generated
CSV exports in `archive/` remain local and ignored; production data belongs under
`data/`. New documents should state their scope and link to their governing
contract or replacement.
