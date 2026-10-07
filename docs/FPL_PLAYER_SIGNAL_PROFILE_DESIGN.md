# FPL Player Signal Profile design

> Status: design/decision record, not a complete inventory of shipped UI.
> Signal cards are implemented in `src/fpl_assistant/apps/viewmodels/player_signal_profile.py`
> and rendered by `apps/fpl/pages/0_main.py`. Unresolved decisions in this draft
> remain unresolved; existing implementation does not approve them implicitly.
> See [application design](FPL_APP_DESIGN.md) for the broader product contract.

## Document control

### Implemented Overview history scope (2026-09-18)

The Overview signal cards use only the season selected in the app. Goal
Scoring, Assist Potential and Defensive Contribution are recalculated from
that season's persisted player match evidence, through the snapshot cutoff.
All eligible season appearances form one window; component rates, position
peer ranks, headline scores, sample sizes and confidence share that scope.
The appearance list now comes from the same selected-season FPL rows used by
FPL Output: at least 30 minutes, including red-card appearances. Provider
statistics join by player and fixture (never gameweek alone). Official FPL
minutes and position own the sample; complete provider coverage is shown
separately for each card. Missing statistics remain unavailable, and missing
core evidence suppresses the headline rather than treating absent rows as zero.
FPL defensive-contribution totals supply threshold hits when available.
Model missing-data and shrinkage rules still apply. No
previous-season fallback or historical trend is shown. Missing season evidence
produces unavailable statistics with the eligible FPL sample still shown.
FPL Output already uses selected-season FPL data; total points and bonus remain
all-appearance season totals, while its rate metrics use eligible appearances.

This is a display-only calculation, cached per artifact version and season.
Published archetypes, Profile details, player labels, historical data, minutes
and points forecasts, and optimizer inputs retain their existing history.
Card scores can therefore differ from the longer-history Profile scores.
Early-season samples have lower confidence and more volatile rankings; players
without eligible season appearances have no production signal yet. The broader
profile modes proposed below remain design material.

| Field | Value |
|---|---|
| Status | Initial design draft |
| Version | 0.2 |
| Last updated | 2026-08-21 |
| Purpose | Working document for review and logic hardening |
| Product surface | FPL application > Players > Profile |
| Primary evidence | V1 archetype snapshots, FPL match history, expected-minutes outputs, expected-points forecasts |

This document proposes a component-based Player Profile for FPL. It is an
initial product and modelling contract, not an approved implementation
specification. Items marked **Open decision** must be resolved before the
relevant feature is treated as production-ready.

### Change history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-08-20 | Initial design evaluation and signal-card proposal. |
| 0.2 | 2026-08-21 | Added decision-hardening workflow, entry templates, worksheets, review gates and append-only decision log. |

## 1. Objective

Create a player profile that explains:

1. Which FPL-relevant signals a player is strong or weak in.
2. Which observed components produce each signal.
3. How the player compares with peers in the same FPL position.
4. How much evidence supports the assessment.
5. Which information is descriptive history and which is a future forecast.

The desired visual hierarchy is:

```text
signal headline -> component evidence -> raw value -> peer percentile
                -> confidence/sample -> technical calculation trail
```

The profile should help a manager understand *why* a player has a particular
classification or projection. It must not collapse unrelated FPL decisions
into an unexplained universal player rating.

## 2. Scope

### 2.1 Initial scope

- Outfield Goal Threat.
- Outfield Assist Potential.
- Position-aware DefCon Potential.
- Usage and minutes security.
- Historical FPL output.
- Historical and forecast value.
- Position-specific clean-sheet, save and goalkeeper signals where supported.
- Raw values, percentiles, confidence, trends and evidence samples.
- A technical audit view backed by persisted model artifacts.

### 2.2 Non-goals

- A generic real-football scouting dashboard.
- A single score intended to identify the universally "best" player.
- Recalculating model scores in the Streamlit presentation layer.
- Treating a percentile as a probability.
- Treating missing provider values as zero.
- Presenting unvalidated descriptive archetypes as future-outcome forecasts.

## 3. Design principles

### 3.1 Separate the semantic scales

The interface must distinguish four kinds of values.

| Value type | Example | Meaning |
|---|---|---|
| Profile percentile | Goal Threat P88 | Stronger historical expression than 88% of eligible same-position peers |
| Raw rate | 0.54 npxG/90 | Observed or model-adjusted historical production rate |
| Absolute probability | 76% start probability | Estimated probability of an event under the stated conditions |
| Expected contribution | 1.8 expected goal points | Forecast FPL points attributable to one scoring component |

Labels, tooltips and formatting must make these scales visually distinct.

### 3.2 Compare like with like

- Production percentiles are calculated within FPL position and season.
- Usage probabilities are absolute calibrated values, not position
  percentiles.
- Forecast probabilities refer to a named fixture or forecast window.
- Double-Gameweek fixtures remain separate match observations.
- Goalkeepers use a separate card scheme from outfield players.

### 3.3 Show both value and context

Every component row should show, where available:

- metric name;
- raw value and unit;
- position percentile;
- direction or interpretation;
- model weight or contribution in a tooltip;
- data status when unavailable.

### 3.4 Preserve evidence honesty

- Missing is displayed as `Unavailable`, never silently converted to zero.
- Provisional and low-confidence signals remain visible but muted.
- Preseason carry-over is labelled as a previous-season baseline.
- Forecast cards are hidden or shown as unavailable when no forecast artifact
  is published.
- Provider and snapshot provenance remain accessible.

## 4. Proposed information architecture

The Profile view contains four levels.

### Level 1: player decision summary

Retain a compact top row containing:

- production composite/archetype;
- usage state;
- profile confidence;
- evidence freshness or model version.

These values orient the user but do not replace the component cards.

### Level 2: mode selector

Proposed modes:

1. **Profile** — temporally combined and shrunk historical signal.
2. **Recent** — latest eligible appearances compared with the historical
   baseline.
3. **Next fixture** — predictive probabilities and expected FPL-point
   components.

The default is **Profile**.

The same card title may appear in more than one mode, but its scale must be
explicit. For example:

```text
Profile:      Goal Threat · P88
Recent:       Recent Goal Threat · P93 · +11 vs baseline
Next fixture: Goal probability · 42%
```

### Level 3: FPL signal cards

Cards appear in a responsive grid:

- two columns on desktop;
- one column on narrow displays;
- no more than six expanded cards in the default view;
- position-irrelevant cards are omitted rather than shown as empty.

### Level 4: technical audit

An optional technical view exposes persisted evidence such as:

- evidence window;
- raw, transformed and winsorized values;
- position z-scores;
- component weights;
- temporal window weights;
- shrinkage;
- final score;
- missing-data flags;
- provenance and model version.

The audit view must not be required to understand the normal profile.

## 5. Card presentation contract

### 5.1 Header

Each profile card should provide:

| Field | Required | Notes |
|---|---:|---|
| Title | Yes | Plain FPL language |
| Headline value | Yes | Percentile, probability, state or expected contribution |
| Scale label | Yes | For example `Position percentile` or `Next-fixture probability` |
| Confidence | When modelled | Insufficient, Low, Medium or High |
| Trend | When modelled | Rising, Stable, Emerging or Declining |
| Evidence sample | For historical cards | Minutes and eligible appearances |
| Forecast context | For forecast cards | GW, opponent, venue and fixture date |

Example:

```text
GOAL THREAT   P88   High confidence   Rising
1,978 minutes · 25 eligible appearances
```

### 5.2 Component row

```text
npxG / 90              0.54     [progress bar]     P91
Shots in box / 90      3.20     [progress bar]     P84
Shots on target / 90   1.31     [progress bar]     P79
Non-penalty goals / 90 0.42     [progress bar]     P72
```

A component progress bar represents the component's peer percentile. It does
not represent the raw value or the component's model weight.

### 5.3 Suggested interpretation bands

| Percentile | Label |
|---:|---|
| 0–19 | Very low |
| 20–39 | Low |
| 40–59 | Average |
| 60–79 | Strong |
| 80–100 | Elite |

The display must not rely on red/green colour alone. Every bar includes a
numeric value and interpretation label.

### 5.4 Tooltips

Each component tooltip should answer:

- What is this metric?
- What is its denominator?
- What peer group produced the percentile?
- Is higher always better?
- What weight does it have in the persisted model?
- Which provider owns the field?

## 6. Signal definitions

### 6.1 Goal Threat

#### Purpose

Describe repeatable non-penalty scoring threat rather than goals alone.

#### Eligible positions

DEF, MID and FWD, ranked separately by FPL position. Goalkeepers are excluded.

#### Headline

Persisted `GOAL_THREAT` score on a 0–100 same-position percentile scale.

#### V1 components

| Component | Current V1 weight | Display unit |
|---|---:|---|
| Non-penalty expected goals | 45% | npxG/90 |
| Shots in the box | 25% | shots/90 |
| Shots on target | 20% | shots/90 |
| Non-penalty goals | 10% | goals/90 |

The UI reads weights and component values from the persisted calculation
artifact. It must not maintain an independent copy of the scoring equation.

#### Naming decision

Use **Goal Threat**, not **Goals**, because the signal combines process and
outcome. Observed goals remain a component and an FPL-output metric.

### 6.2 Assist Potential

#### Purpose

Describe repeatable chance creation relevant to future FPL assists.

#### Eligible positions

DEF, MID and FWD, ranked separately by FPL position. Goalkeepers are excluded
unless a later validated goalkeeper-creation signal is introduced.

#### Headline

Persisted `CREATOR` score on a 0–100 same-position percentile scale.

#### V1 components

| Component | Current V1 weight | Display unit |
|---|---:|---|
| Expected assists | 50% | xA/90 |
| Key passes | 25% | key passes/90 |
| Big chances created | 15% | chances/90 |
| Shot-creating actions | 10% | actions/90 |

#### Naming decision

Use **Assist Potential**, with `Creator` retained as the archetype label. Raw
assists should not be treated as the complete underlying signal.

### 6.3 DefCon Potential

#### Purpose

Explain a player's ability to accumulate the actions that produce FPL
defensive-contribution points.

#### Eligible positions and thresholds

- DEF: at least 10 CBIT actions in a match.
- MID/FWD: at least 12 CBIRT actions in a match.
- GKP: not eligible for outfield defensive-contribution points.

#### Headline

The initial Profile-mode headline is the persisted `DEFENSIVE_ENGINE` score.
The card must also display the actual historical DefCon hit rate because that
has direct FPL meaning.

#### Components

- DefCon hit rate.
- Tackles won.
- Interceptions.
- Clearances.
- Blocks.
- Recoveries for MID/FWD.

The UI uses the component weights stored in the evidence ledger. Position-
inapplicable components are omitted, and missing provider fields are not
treated as zero.

#### Open decision

Decide whether the user-facing headline should remain the broader Defensive
Engine percentile or become a purpose-built forecast of `P(DefCon points)`.
Until a DefCon probability model is validated, the historical percentile and
historical hit rate must remain separately labelled.

### 6.4 Usage

#### Purpose

Describe current minutes security and substitution risk when the player is
available.

#### Headline

Mutually exclusive state:

- Nailed;
- Regular Starter;
- Rotation Risk;
- Impact Sub;
- Fringe.

#### Components

| Component | Scale |
|---|---|
| Start probability | 0–100% absolute probability |
| Expected minutes | 0–90 minutes per available team match |
| Cameo probability | `P(cameo | not starting, available)` |
| 60-minute probability | 0–100% absolute probability |
| Availability | Available, doubtful, injured or suspended |

Usage must not be labelled as a peer percentile. Confirmed unavailable matches
are excluded from the start-rate denominator according to the usage contract,
with confidence reduced where appropriate.

#### Open decision

Expected-minutes V2 must define which published fields are canonical for this
card and how its uncertainty interval is displayed. The Profile must not bind
permanently to experimental column names.

### 6.5 FPL Output

#### Purpose

Show realised FPL production without presenting it as underlying ability.

#### Proposed components

- Total points.
- Points per appearance.
- Points per 90.
- Return rate.
- Haul probability/rate.
- Blank rate.
- Bonus points or bonus frequency.
- Position-relevant clean-sheet, save and DefCon points.

#### Headline

**Open decision:** choose between position percentile of points per eligible
appearance and an established Return Shape score. Do not average raw totals
with rates.

### 6.6 Value

#### Purpose

Separate production quality from price efficiency.

#### Historical mode

- Current or snapshot price, explicitly labelled.
- Historical points per million.
- Position price percentile.
- Historical value-over-replacement percentile.

#### Forecast mode

- Forecast xPts.
- Forecast xPts per million.
- Forecast window and number of fixtures.
- Ownership as context, not as a quality component.

Historical points/£m and forecast xPts/£m must not share an ambiguous `Value`
label.

### 6.7 Clean-sheet and defensive-return opportunity

#### Eligible positions

GKP and DEF by default. MID clean-sheet points may be shown in the expected-
points breakdown but should not receive a defender-style defensive profile.

#### Proposed components

- Historical clean-sheet rate in eligible appearances.
- Team clean-sheet probability for the next fixture.
- Team expected goals conceded.
- Player 60-minute probability.
- Expected clean-sheet FPL points.

This card must distinguish team defensive strength from player usage.

### 6.8 Goalkeeper scheme

Goalkeepers do not receive the outfield Goal Threat, Assist Potential or
DefCon cards.

Proposed goalkeeper cards:

1. **Shot stopping** — save percentage, saves/90 and goals prevented where
   sufficiently covered.
2. **Save potential** — shots on target faced, saves and expected save points.
3. **Clean-sheet outlook** — team defensive forecast combined with P(60+).
4. **Usage** — start probability and expected minutes.
5. **Value** — historical and forecast points per million.

Sweeping and distribution remain descriptive football traits unless a tested
link to future FPL returns justifies their prominence.

## 7. Profile, recent and forecast modes

### 7.1 Profile mode

- Source: persisted archetype snapshot and calculation evidence.
- Window: model-defined temporal combination.
- Adjustment: winsorization, position standardization and shrinkage as stored.
- Headline: descriptive position percentile or state.
- Default availability: available whenever a complete snapshot exists.

### 7.2 Recent mode

- Source: persisted recent-window evidence.
- Headline: recent percentile and change from stored baseline.
- Trend: use persisted trend and persistence rules.
- Minimum evidence: follow the archetype specification.
- Do not let an arbitrary UI window silently replace the model-defined recent
  window.

#### Open decision

Decide whether users may select alternative windows such as last 5 or last 10.
If enabled, label them as exploratory summaries rather than archetype scores.

### 7.3 Next-fixture mode

- Source: published expected-points forecast artifact.
- Context: named GW, opponent, venue and scheduled date.
- Components may include:
  - predicted minutes;
  - goal probability;
  - assist probability;
  - team clean-sheet probability;
  - DefCon-points probability;
  - expected appearance points;
  - expected goal points;
  - expected assist points;
  - expected clean-sheet points;
  - expected save points;
  - expected DefCon points;
  - total xPts.

If no current forecast is published, show an explicit unavailable state. Never
substitute a previous-season forecast.

## 8. Confidence, trends and missing data

### 8.1 Confidence

Display the persisted confidence band and, where helpful, its numeric value.

| Band | Display treatment |
|---|---|
| Insufficient | Suppress firm interpretation; explain missing evidence |
| Low | Muted card/header and provisional wording |
| Medium | Normal card with confidence qualifier |
| High | Normal card with strong evidence label |

Confidence is evidence adequacy, not the probability that the headline is
correct.

### 8.2 Trends

Use persisted trend states rather than deriving a direction from a single
snapshot in the UI. Show the score delta only when the baseline and recent
values are compatible.

### 8.3 Missing values

For each unavailable component, display one of:

- Provider data unavailable.
- Core field missing.
- Insufficient minutes.
- Insufficient appearances.
- Position not applicable.
- Forecast not published.

Do not render a percentile bar for unavailable evidence.

## 9. Data and integration contract

### 9.1 Existing sources

| Need | Existing source |
|---|---|
| Profile headlines | `archetypes.jsonl` |
| Production components | `production_component_evidence` |
| Usage and other families | `family_calculation_evidence` |
| Technical match evidence | `player_match_evidence` |
| Legacy three-dimension profile | `analytics/player_profiles.csv` |
| Match-level FPL facts | `gws/merged_gws.csv` |
| Usage model outputs | `data/models/minutes/expected_minutes.csv` or its approved V2 successor |
| Next-fixture probabilities and xPts components | Published expected-points forecast |
| Set-piece roles | WhoScored set-piece role artifact |

### 9.2 Proposed presentation view model

The Streamlit page should consume a stable structure rather than parse model
artifacts directly.

```python
{
    "player_id": "...",
    "mode": "profile",
    "as_of": "...",
    "cards": [
        {
            "id": "goal_threat",
            "title": "Goal threat",
            "headline_value": 88.0,
            "headline_type": "position_percentile",
            "confidence_band": "High",
            "trend": "Rising",
            "evidence_minutes": 1978,
            "eligible_appearances": 25,
            "components": [
                {
                    "id": "npxg",
                    "label": "Non-penalty xG / 90",
                    "raw_value": 0.54,
                    "unit": "per 90",
                    "percentile": 91.0,
                    "weight": 0.45,
                    "status": "available",
                    "provider": "understat",
                }
            ],
        }
    ],
}
```

### 9.3 Presentation rules

- The page does not recompute scores.
- Component weights come from the stored evidence artifact.
- Display names and definitions come from a versioned metric catalogue.
- The view model owns units, formatting and missing-state mapping.
- Streamlit owns layout and interaction only.

## 10. Performance design

The current Profile flow can load the complete match-evidence JSONL before
filtering it to one player. The design should avoid making this cost part of
the default card experience.

Recommended approach:

1. Read headline and component-summary artifacts first.
2. Prefer Parquet projection and player filters for evidence tables.
3. Do not load match-level evidence until the user explicitly enables the
   technical audit.
4. Cache by artifact path, modification time and size.
5. Consider a small per-player display artifact if predicate filtering is not
   reliable in the deployed environment.

**Open decision:** Streamlit expanders execute their contents even when
collapsed. Use an explicit checkbox, toggle or segmented control to gate
loading the technical evidence.

## 11. Validation and release rules

### 11.1 Descriptive cards

Verify:

- score parity with the persisted snapshot;
- component parity with the evidence ledger;
- correct same-position reference population;
- correct units and denominators;
- missing-data behaviour;
- confidence and trend parity;
- no future leakage in evidence selection.

### 11.2 Forecast cards

Forecast probabilities require their own validation. At minimum:

- Brier score and calibration for start, goal, assist, clean-sheet and DefCon
  events where applicable;
- MAE for expected minutes;
- expected-points component reconciliation;
- chronological out-of-sample evaluation;
- calibration reporting by position and evidence level.

The current archetype validation result does not clear the documented
production release bar. Until the relevant targets pass, archetype scores must
be presented as descriptive profiles rather than claims about future returns.

## 12. Accessibility and responsive behaviour

- Never encode quality or risk by colour alone.
- Always show raw values and `Pxx`, percentage or state labels.
- Use sufficient text/background contrast.
- Add definitions for abbreviations such as npxG, CBIT and CBIRT.
- Preserve logical keyboard order through card and component controls.
- Collapse to one card per row on mobile.
- Avoid horizontal scrolling within a component card.
- Use a readable proportional UI font; reserve monospaced text for technical
  evidence only.

## 13. Delivery plan

### Phase 1: evidence-backed MVP

- Introduce the stable card view model.
- Implement Goal Threat, Assist Potential, DefCon Potential and Usage.
- Display raw values, percentiles/probabilities, confidence, trend and sample.
- Preserve the existing calculation audit behind an explicit control.
- Add position-specific omission rules.

### Phase 2: FPL decision layer

- Add FPL Output and Value.
- Add Recent mode.
- Add Next-fixture mode from the published forecast.
- Add expected-points component reconciliation.

### Phase 3: position and comparison depth

- Add the goalkeeper scheme.
- Add clean-sheet opportunity.
- Add set-piece routes where confidence is adequate.
- Add comparison overlays for shortlisted players.
- Add historical card trends across snapshots.

## 14. Initial acceptance criteria

The initial implementation is acceptable when:

1. Every displayed headline identifies its scale.
2. Historical production scores exactly match the latest selected snapshot.
3. Every component raw value and model weight matches the stored calculation
   evidence.
4. Percentiles use the correct FPL-position reference population.
5. Usage values are displayed as absolute probabilities/minutes, not
   percentiles.
6. DefCon rules differ correctly between DEF and MID/FWD, and exclude GKP.
7. Missing core evidence produces an unavailable state rather than zero.
8. Previous-season carry-over is explicitly labelled.
9. Forecast mode never falls back silently to another season.
10. The layout works at desktop and narrow widths without horizontal card
    scrolling.
11. Technical evidence is not loaded during the default Profile render.
12. Unit, view-model and Streamlit smoke tests cover each position scheme and
    evidence state.

## 15. Open decisions register

| ID | Status | Question | Initial recommendation | Evidence required |
|---|---|---|---|---|
| PROFILE-01 | Draft | Should there be an overall player score? | No. Retain separate decision signals. | Decision-use analysis showing whether an aggregate improves transfer/captain decisions without hiding trade-offs. |
| PROFILE-02 | Draft | Should DefCon headline be descriptive or predictive? | Show Defensive Engine percentile plus historical hit rate until `P(DefCon points)` is validated. | Chronological calibration and discrimination results for a DefCon-points probability model. |
| PROFILE-03 | Draft | What is the FPL Output headline? | Evaluate position percentile of points per eligible appearance versus Return Shape score. | Stability, predictive validity and user-interpretability comparison. |
| PROFILE-04 | Draft | Can users select arbitrary recent windows? | Allow exploratory summaries, but keep model scores tied to model-defined windows. | UX test plus leakage and semantic review of selectable windows. |
| PROFILE-05 | Draft | Which expected-minutes V2 fields are canonical? | Resolve in the V2 implementation contract before binding the UI. | Approved V2 schema, units, calibration contract and migration mapping. |
| PROFILE-06 | Draft | How should uncertainty be shown? | Prefer confidence band plus interval where the underlying model publishes one. | Coverage tests, interval interpretation and compact-layout evaluation. |
| PROFILE-07 | Draft | Should forecast component bars be probabilities or expected points? | Use probabilities for event cards and expected points for the xPts breakdown; do not combine their scales. | Reconciliation test and comprehension review using realistic fixtures. |
| PROFILE-08 | Draft | When should set-piece roles become a card? | Only when role, share and recency confidence are available. | Role coverage, stability and stale-role detection results. |
| PROFILE-09 | Draft | Should the legacy profile remain visible? | Use only as a labelled fallback when no V1 snapshot exists. | Coverage audit and migration/deprecation impact assessment. |
| PROFILE-10 | Draft | How should transferred players be presented? | Retain carried evidence, show Provisional, and expose new-club sample/confidence. | Transfer backtest and tests of reset, carry-over and confidence recovery. |

Allowed decision statuses are:

```text
Draft -> Evidence gathering -> Proposed -> Accepted / Rejected / Deferred
      -> Implemented -> Verified
```

`Implemented` does not mean `Verified`. A decision becomes Verified only after
its stated tests and release evidence pass.

## 16. Decision-hardening workflow

Every material product, metric or modelling choice should follow the same
workflow.

### 16.1 Frame the decision

- Assign a stable decision ID.
- State one answerable question.
- Identify the affected cards, modes, positions and artifacts.
- Explain why the choice matters to an FPL decision.
- Define what is explicitly outside the decision's scope.

### 16.2 Establish invariants

Record the rules that no option may violate. Typical invariants include:

- no future leakage;
- no hidden prior-season fallback;
- no missing-to-zero coercion;
- no percentile/probability ambiguity;
- correct position-specific FPL rules;
- persisted calculation parity;
- versioned model-definition changes.

### 16.3 Compare viable options

Include the status quo as an option. For each option, assess:

- semantic correctness;
- decision usefulness;
- data availability and provenance;
- statistical validity and calibration;
- sample-size behaviour;
- implementation and maintenance cost;
- performance impact;
- accessibility and comprehension risk;
- migration and rollback requirements.

### 16.4 Define evidence before choosing

Specify the evidence capable of accepting or rejecting an option before
reviewing its final results. Evidence may include:

- chronological backtests;
- calibration and error metrics;
- artifact coverage audits;
- parity tests against persisted ledgers;
- small-sample and missing-data simulations;
- user comprehension tests;
- render-time and memory measurements;
- desktop and mobile visual review.

### 16.5 Record the decision

The entry must identify the selected option, rejected alternatives, rationale,
consequences and unresolved risks. Avoid recording only the conclusion.

### 16.6 Implement and verify

- Link the decision to its implementation and tests.
- Verify all acceptance conditions.
- Record deviations discovered during implementation.
- Update the decision if evidence changes; do not silently change the logic.

## 17. General decision entry template

Copy this section for each detailed decision. Use the stable ID from the open
decisions register or create a new ID in the appropriate namespace.

### `[DECISION-ID] — Short decision title`

#### Decision metadata

| Field | Entry |
|---|---|
| Status | Draft |
| Owner | TBD |
| Reviewers | TBD |
| Date opened | YYYY-MM-DD |
| Date decided | TBD |
| Target version | TBD |
| Affected positions | GKP / DEF / MID / FWD / All |
| Affected modes | Profile / Recent / Next fixture |
| Related documents | TBD |

#### Question

State the single question this entry resolves.

#### Decision context

Describe the current behaviour, the user need and the failure mode or ambiguity
that makes a decision necessary.

#### Scope

**In scope**

- TBD

**Out of scope**

- TBD

#### Invariants

- [ ] No future leakage.
- [ ] Missing evidence is not converted to zero.
- [ ] Percentiles, probabilities and expected values remain distinguishable.
- [ ] Position-specific FPL rules are preserved.
- [ ] Presentation values reconcile with their canonical artifact.
- [ ] Any model-definition change receives a new version.

Add decision-specific invariants below these common requirements.

#### Options considered

| Option | Description | Advantages | Disadvantages | Evidence status |
|---|---|---|---|---|
| A | Status quo | TBD | TBD | TBD |
| B | Proposed alternative | TBD | TBD | TBD |
| C | Additional viable alternative | TBD | TBD | TBD |

#### Evidence plan

| Evidence | Dataset/artifact | Metric or test | Acceptance threshold | Result |
|---|---|---|---|---|
| Coverage | TBD | Non-null and eligible-player coverage | TBD | Pending |
| Validity | TBD | Chronological out-of-sample result | TBD | Pending |
| Calibration | TBD | Brier/ECE or interval coverage | TBD | Pending |
| Parity | TBD | Stored-artifact reconciliation | Exact | Pending |
| UX | TBD | Comprehension/task result | TBD | Pending |
| Performance | TBD | Render time and peak memory | TBD | Pending |

Use `Not applicable` with a reason rather than deleting an evidence row.

#### Decision

**Selected option:** TBD

**Decision statement:** TBD

#### Rationale

Explain why the selected option best satisfies the invariants and acceptance
thresholds. Identify important evidence against the decision as well as in its
favour.

#### Rejected alternatives

| Option | Reason rejected | Reconsider when |
|---|---|---|
| TBD | TBD | TBD |

#### Consequences and risks

**Positive consequences**

- TBD

**Negative consequences/trade-offs**

- TBD

**Residual risks**

- TBD

#### Implementation contract

| Concern | Required change |
|---|---|
| Canonical data fields | TBD |
| View-model contract | TBD |
| UI behaviour | TBD |
| Missing-data behaviour | TBD |
| Migration/backfill | TBD |
| Documentation | TBD |

#### Verification

- [ ] Unit tests added.
- [ ] Artifact parity tests added.
- [ ] Position-specific edge cases covered.
- [ ] Missing and insufficient-evidence states covered.
- [ ] Streamlit smoke test passed.
- [ ] Desktop and narrow-width visual review passed.
- [ ] Performance threshold passed.
- [ ] Documentation and decision register updated.

#### Rollback and supersession

Define how the change can be disabled or rolled back, which artifacts remain
compatible and which later decision would supersede this one.

## 18. Metric and card hardening worksheet

Complete one worksheet for every headline signal and every component metric.
This is narrower than a general decision entry and is intended to prevent
under-specified display logic.

### 18.1 Card-level worksheet

| Field | Required entry |
|---|---|
| Card ID and title | Stable machine ID and user-facing name |
| User decision supported | Transfer, captaincy, benching, comparison or monitoring |
| Eligible positions | Exact FPL positions |
| Eligible population | Exact minutes/appearance/status rules |
| Headline definition | Formula or canonical stored field |
| Headline scale | Percentile, probability, expected value, state or raw rate |
| Reference population | Position, season and eligibility filters |
| Evidence window | Historical, recent or named forecast window |
| Confidence source | Canonical field/formula and display bands |
| Trend source | Canonical field/formula and persistence rule |
| Missing-core behaviour | Suppress, provisional or unavailable |
| Missing-secondary behaviour | Reweighting rule and confidence penalty |
| Position-specific variants | Exact differences by position |
| Validation target | Future outcome or descriptive parity target |
| Release threshold | Quantified go/no-go rule |
| Canonical artifacts | Paths/logical tables, not presentation copies |

### 18.2 Component-level worksheet

| Field | Required entry |
|---|---|
| Component ID and label | Stable ID and plain-language label |
| Definition | Provider-aware canonical definition |
| Source owner | FPL, WhoScored, Understat or derived model |
| Numerator | Exact event/value definition |
| Denominator | Per 90, appearance, start, available team match or event opportunity |
| Direction | Higher, lower or contextual |
| Transformation | Log, cap, winsorization or none |
| Standardization | Position/season group and method |
| Weight | Persisted model weight or not applicable |
| Display unit and precision | Exact UI formatting |
| Tooltip explanation | User-facing interpretation |
| Missing-value rule | Exact behaviour and reason code |
| Edge cases | Red cards, DGWs, transfers, position changes, partial matches |
| Tests | Boundary, parity, missingness and denominator tests |

### 18.3 Forecast-specific worksheet

| Field | Required entry |
|---|---|
| Forecast target | Exact future event/value |
| Prediction horizon | Named fixture, GW or multi-GW window |
| Conditioning information | Availability, opponent, venue and known schedule state |
| Calibration population | Position/evidence strata |
| Probability interpretation | Conditional or unconditional statement |
| Point reconciliation | How the component contributes to total xPts |
| Uncertainty | Interval/distribution and coverage target |
| Baselines | Required naive and incumbent comparators |
| Leakage controls | Time cutoff and feature-availability checks |
| Release evidence | Chronological folds and acceptance results |

## 19. Review gates

### Gate A: definition ready

- The question, scope and invariants are complete.
- All viable options are represented fairly.
- Canonical fields and denominators are identified.
- Acceptance thresholds are defined before final results are reviewed.

### Gate B: evidence ready

- Coverage and provenance audits pass.
- Required backtests, parity tests and edge-case simulations are complete.
- Negative or contradictory evidence is recorded.
- Results are reproducible from versioned artifacts.

### Gate C: decision accepted

- The selected option and rationale are explicit.
- Rejected options and reconsideration triggers are recorded.
- Residual risks have owners or explicit acceptance.
- Model/data contract changes have a versioning plan.

### Gate D: implementation ready

- View-model and UI contracts are defined.
- Missing, provisional and unavailable states are specified.
- Migration, rollback and performance plans are defined.
- Required tests are enumerated.

### Gate E: verified

- Implementation reconciles with the accepted decision.
- Automated and visual checks pass.
- Performance stays within the agreed budget.
- Documentation, register status and evidence links are current.

## 20. Decision log

Add one row whenever a decision changes status. Do not rewrite history; append a
new row that references the superseded entry.

| Date | Decision ID | From status | To status | Summary | Evidence/implementation reference | Author |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | PROFILE-XX | Draft | Proposed | Example placeholder; replace when the first decision is hardened. | TBD | TBD |

## 21. Logic-hardening checklist

Before implementation approval, resolve and document:

- [ ] Exact headline definition for every card.
- [ ] Exact reference population for every percentile.
- [ ] Exact denominator for every raw rate.
- [ ] Metric direction and interpretation.
- [ ] Canonical artifact and field for every displayed value.
- [ ] Missing-core and missing-secondary behaviour.
- [ ] Minimum minutes, appearances and starts.
- [ ] Shrinkage and temporal-window source.
- [ ] Transfer, position-change and promoted-player behaviour.
- [ ] Injury and suspension denominator rules.
- [ ] Confidence and uncertainty presentation.
- [ ] Forecast calibration and release requirements.
- [ ] Position-specific card ordering.
- [ ] Goalkeeper fallback behaviour.
- [ ] Tooltip metric catalogue and provider provenance.
- [ ] Performance budget and lazy-loading mechanism.
- [ ] Mobile and accessibility verification.

## 22. Relationship to existing documents

This draft is subordinate to the versioned model and data contracts. If it
conflicts with them, the model/data contract governs until an explicit version
change is approved.

Relevant documents:

- `docs/FPL_ARCHETYPE_V1.md`
- `docs/fpl_archetype_scheme_catalogue.md`
- `docs/FPL_APP_DESIGN.md`
- `docs/FPL_PIPELINE.md`
- `docs/FPL_expected_minutes_model_v2_design.md`
- `docs/PLAYER_PROFILES.md`
