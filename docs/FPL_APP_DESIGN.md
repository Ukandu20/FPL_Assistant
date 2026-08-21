# FPL application design

## Product purpose

The Streamlit application is a focused FPL decision-support product. It helps
users discover, compare, shortlist, and understand players using current FPL
facts, forecasts when published, historical baselines, fixtures, team context,
and versioned archetype evidence.

Provider-specific research interfaces are deliberately outside the public app.
Provider data may support an FPL decision, but the interface describes the FPL
meaning rather than exposing a separate provider workflow.

## Navigation

The canonical entry point is:

```powershell
streamlit run apps/fpl/app.py
```

The registered pages are:

1. **Gameweek Hub** — default decision centre, fixture swings, risks, archetype
   changes, candidates, and shortlist.
2. **Players** — discovery filters and a four-view Player Card.
3. **Compare** — persistent comparison for two to four players.
4. **Teams** — fixture context, squad archetypes, set pieces, and player links.
5. **League** — one-league standings and underlying team context.

## Evidence rules

- A missing forecast is displayed as missing; another season is never used as
  a hidden projection fallback.
- Preseason rankings use a clearly labelled previous-season baseline.
- Official deadlines come from the persisted FPL gameweek calendar. When that
  artifact is absent, the Hub displays a clearly labelled estimate calculated
  as the first scheduled kickoff minus the official 90-minute cutoff.
- Stat leaders are calculated from match-level FPL rows. Rate tables expose
  their appearance or minute denominator and never use a hidden prior-season
  fallback.
- Low-confidence archetype labels are visually muted but not silently hidden.
- Team performance and set-piece baselines identify their evidence season.

## Shared state

Shareable context uses query parameters:

```text
/players?season=2026-2027&player=<id>&view=Overview
/teams?season=2026-2027&team=ARS
/compare?season=2026-2027&players=<id1>,<id2>
/league?season=2025-2026
```

Session state owns the shortlist, comparison selection, recent players, and
discovery preferences. Table selections connect League to Teams and Teams or
Hub results to Player Cards.

## Visual system

`.streamlit/config.toml` supplies the base theme. `apps/fpl/ui.py` owns shared
page headers, empty states, semantic colours, badges, freshness labels, chart
styling, responsive spacing, and metric-card treatment.

The interface uses no more than four metric cards per row. Charts use labels
and axes in addition to colour, and diverging charts use a brown/blue-green
scale rather than red/green-only encoding.

## Performance

The Player and League pages use conditional view controls instead of eagerly
rendering every analytical section. Large archetype calculation ledgers are
loaded only for the Profile view. Data loaders are keyed by file modification
time and size so Streamlit caching remains fresh without reading every file on
every rerun.

## Verification

Pure ranking, fixture, archetype, comparison, and preseason fallback behaviour
is tested in `tests/test_fpl_dashboard_viewmodels.py`. Existing Player Card and
profile tests cover compatibility. Every registered page should also complete
a Streamlit `AppTest` smoke run with no exceptions for its default local data
state.
