# FPL Expected-Minutes Model V2 — Implementation Hardening Decision Contract

> Status: governing V2.0 implementation contract. This takes precedence over
> the earlier design and completed-defaults documents. Operational commands
> are maintained in [V2 operations](FPL_EXPECTED_MINUTES_V2.md).
> Implementation readiness does not imply production acceptance.

## Purpose

This document is the final decision layer for the **FPL Expected-Minutes Model V2**.

The V2 architecture is already conceptually defined. This contract exists to remove the remaining implementation ambiguity so that, once the decisions below are filled in, two developers should be able to implement the same system without making different modelling assumptions.

For every decision:

- exactly three options are provided;
- one option is marked **Recommended default**;
- the trade-offs are stated;
- a `Final decision` field is provided for the implementation owner to complete.

The design should not be treated as implementation-ready until all **Required before V2.0 implementation** decisions are completed.

## Contract status

**Status: HARDENED — V2.0 implementation contract complete.**

All implementation-critical decisions are now locked. `team_rot3` is excluded from V2.0, walk-forward folds are deterministic, calibration fallback thresholds are numeric, and production non-inferiority is defined with paired block-bootstrap confidence intervals and practical margins. This document can now be treated as the canonical V2.0 implementation contract unless explicitly versioned and changed.


---

# 1. V2.0 Scope Boundary

**Decision:** How much of the broader V2 design should be included in the first implementation?

### Option A — Minimal architecture migration **(Recommended default)**

Implement only:

- state decomposition;
- soft expected-minutes mixture;
- simplified position pooling;
- probability calibration selection;
- native missing-value handling;
- evidence-strength features;
- improved evaluation and instrumentation.

Defer:

- previous-season priors;
- injury/status data;
- fixture congestion;
- manager effects;
- player-specific rotation features.

**Strengths**

- isolates whether the architecture itself improves the model;
- fastest path to a clean V2 baseline;
- lowest leakage/data-engineering risk;
- easiest comparison against V1.

**Weaknesses**

- does not solve early-season cold-start problems immediately;
- does not yet use potentially powerful availability information.

### Option B — Architecture + season priors

Implement Option A plus:

- previous-season start prior;
- previous-season minutes prior;
- prior shrinkage for GW1+.

**Strengths**

- directly addresses early-season role resets;
- improves usefulness for GW1–5.

**Weaknesses**

- introduces several new hyperparameters;
- harder to determine whether gains come from architecture or priors.

### Option C — Full contextual V2

Implement architecture, priors, availability, congestion, transfers, manager changes, and player-specific rotation together.

**Strengths**

- closest to the eventual ideal production system.

**Weaknesses**

- highest leakage risk;
- hardest to debug;
- makes ablation attribution weak;
- substantially increases implementation complexity.

**Final decision:** Option A — Minimal architecture migration. V2.0 includes state decomposition, pure soft mixture, simplified pooling, calibration selection, native missing values, evidence-strength features, and improved evaluation/instrumentation. Season priors, availability, congestion, manager effects, and player-specific rotation are deferred.

---

# 2. Position Pooling Strategy

**Decision:** How should position-specific models be structured in V2.0?

### Option A — One global model

Train one model per head across:

- GK;
- DEF;
- MID;
- FWD.

Position is supplied as a categorical feature.

**Strengths**

- maximum training sample;
- simplest maintenance;
- strongest statistical pooling.

**Weaknesses**

- goalkeeper minutes behaviour is structurally different from outfield players.

### Option B — GK separate + pooled outfield **(Recommended default)**

Train:

- one GK model family;
- one combined DEF/MID/FWD model family.

Outfield position remains a categorical feature.

**Strengths**

- isolates the genuinely different goalkeeper process;
- preserves large outfield sample;
- lower complexity than four independent families.

**Weaknesses**

- assumes DEF/MID/FWD differences can be learned through features.

### Option C — Fully position-specific

Train separate model families for:

- GK;
- DEF;
- MID;
- FWD.

**Strengths**

- maximum position specialization;
- can capture structurally different rotation/substitution patterns.

**Weaknesses**

- smaller training sets;
- more calibrators;
- more overfitting risk;
- more artifacts and maintenance.

**Final decision:** Option B — GK separate + pooled outfield. Use one goalkeeper model family and one combined DEF/MID/FWD model family.

---

# 3. Position Encoding

**Decision:** How should position be represented inside pooled models?

### Option A — Ordinal integer encoding

Example:

```text
GK=0
DEF=1
MID=2
FWD=3
```

**Strengths**

- simple;
- compatible with current code.

**Weaknesses**

- implies artificial numeric ordering and distance.

### Option B — Native/categorical encoding **(Recommended default)**

Treat `pos` as a true categorical feature.

**Strengths**

- no false numerical ordering;
- allows nonlinear category-specific effects;
- most conceptually correct.

**Weaknesses**

- implementation depends on chosen LightGBM data interface.

### Option C — One-hot encoding

Create:

- `pos_DEF`;
- `pos_MID`;
- `pos_FWD`;
- etc.

**Strengths**

- explicit and portable;
- easy to audit.

**Weaknesses**

- slightly wider feature matrix;
- redundant if GK already has its own model.

**Final decision:** Option B — Native/categorical encoding. Treat `pos` as categorical, not ordinal.

---

# 4. Start-Probability Training Population

**Decision:** What population should define the target for `P(start)`?

### Option A — All registered club players

Every registered player receives:

\[
Start \in \{0,1\}
\]

for every fixture.

**Strengths**

- directly estimates absolute starting probability.

**Weaknesses**

- injuries, suspensions and players outside matchday consideration become mixed with tactical role;
- requires very reliable roster history.

### Option B — Fixture-eligible players **(Recommended default)**

Train only players who were legitimately part of the club's eligible selection universe for the target fixture.

Known pre-deadline unavailable players are handled separately or flagged.

**Strengths**

- cleaner tactical-role target;
- avoids teaching the model that injury absence equals rotation;
- aligns well with eventual availability modelling.

**Weaknesses**

- requires an explicit eligibility definition.

### Option C — Matchday squad only

Train only players listed in the matchday squad.

**Strengths**

- cleanest tactical starting-vs-bench comparison.

**Weaknesses**

- impossible to forecast whether a player makes the squad;
- overly optimistic production universe;
- introduces selection bias.

**Final decision:** Option B — Fixture-eligible players. Train `P(start)` only on players legitimately eligible for selection for the fixture.

---

# 5. Fixture Eligibility Definition

**Decision:** How should `eligible_for_fixture` be defined for V2.0?

### Option A — Roster membership only

Eligible if the player belongs to the club on the fixture date.

**Strengths**

- easy to construct;
- broad coverage.

**Weaknesses**

- includes known injured/suspended players.

### Option B — Roster + known pre-deadline availability **(Recommended default)**

Eligible if:

- player belongs to club;
- no confirmed pre-deadline suspension/absence excludes them;
- player has not transferred away;
- source data indicates legitimate squad availability.

**Strengths**

- better definition of actual selection opportunity;
- prevents obvious availability contamination.

**Weaknesses**

- requires timestamp-safe status information.

### Option C — Historical fixture-calendar presence

Eligible only if the source fixture calendar already contains the player row.

**Strengths**

- easiest to reproduce from existing pipeline.

**Weaknesses**

- depends heavily on how the calendar was generated;
- can silently exclude valid zero-minute players.

**Final decision:** Option B — Roster + known pre-deadline availability. Eligible means registered with the club and not confirmed, before the prediction cutoff, to be suspended, transferred away, or definitely unavailable.

---

# 6. Start Head Feature Set

**Decision:** What should the baseline `P(start)` feature list contain?

### Option A — Minimal core

```text
min_lag1
min_ewm_hl2
start_lag1
start_rate_hl3
pos
history_matches
```

**Strengths**

- extremely interpretable;
- strong baseline;
- minimal redundancy.

**Weaknesses**

- may react slowly to role transitions.

### Option B — Core + role stability **(Recommended default)**

```text
min_lag1
min_ewm_hl2
start_lag1
start_rate_hl3
start_streak
bench_streak
days_feat
history_matches
pos
```

**Strengths**

- balances recent shocks and established role;
- preserves useful existing features;
- still relatively compact.

**Weaknesses**

- some correlation/redundancy remains.

### Option C — Expanded current-context baseline

Option B plus:

```text
played_last
long_gap14
team_rot3
fdr
```

**Strengths**

- preserves most current information.

**Weaknesses**

- weaker causal justification for some variables;
- harder baseline to interpret;
- less clean for ablation.

**Final decision:** Option B — Core + role stability.

```python
START_FEATURES = [
    "min_lag1",
    "min_ewm_hl2",
    "start_lag1",
    "start_rate_hl3",
    "start_streak",
    "bench_streak",
    "days_feat",
    "history_matches",
    "pos",
]
```

---

# 7. Starter-Minutes Head Feature Set

**Decision:** What should predict:

\[
E[M \mid Start]
\]

?

### Option A — Minutes-history only

```text
min_lag1
min_ewm_hl2
history_matches
pos
```

**Strengths**

- extremely direct;
- likely captures substitution tendency efficiently.

**Weaknesses**

- ignores role stability and schedule context.

### Option B — Minutes + recent starting role **(Recommended default)**

```text
min_lag1
min_ewm_hl2
start_rate_hl3
days_feat
history_matches
pos
```

**Strengths**

- still simple;
- lets duration vary with recent role strength;
- useful for testing whether uncertain starters play fewer minutes.

**Weaknesses**

- some overlap between start rate and minutes history.

### Option C — Same feature set as start classifier

Use all `P(start)` features.

**Strengths**

- simplest feature-management implementation.

**Weaknesses**

- assumes every start-probability feature is relevant to conditional duration;
- may unnecessarily couple separate modelling questions.

**Final decision:** Option B — Minutes + recent starting role.

```python
START_MIN_FEATURES = [
    "min_lag1",
    "min_ewm_hl2",
    "start_rate_hl3",
    "days_feat",
    "history_matches",
    "pos",
]
```

---

# 8. Cameo Probability Training Population

**Decision:** What should `P(cameo)` mean?

### Option A — `P(cameo | not started)`

All non-starting eligible players enter the cameo classifier.

**Strengths**

- directly fits the state-tree formula;
- simple.

**Weaknesses**

- some non-starters may not have been named on the bench.

### Option B — `P(cameo | available and not started)` **(Recommended default)**

Use non-starting players who were legitimately available for selection.

**Strengths**

- avoids mixing absence with substitution decisions;
- coherent with fixture eligibility.

**Weaknesses**

- depends on availability quality.

### Option C — `P(cameo | named substitute)`

Use only actual bench players.

**Strengths**

- very clean substitution model.

**Weaknesses**

- requires another model for probability of making the bench;
- incomplete state tree for current V2.

**Final decision:** Option B — `P(cameo | available and not started)`. Use eligible non-starters rather than only named substitutes.

---

# 9. Cameo Probability Feature Set

**Decision:** What should predict cameo probability?

### Option A — Minimal bench history

```text
min_lag1
min_ewm_hl2
bench_streak
pos
history_matches
```

**Strengths**

- simple;
- directly related to substitute role.

**Weaknesses**

- ignores recent starting-role strength.

### Option B — Bench + role history **(Recommended default)**

```text
min_lag1
min_ewm_hl2
start_rate_hl3
bench_streak
days_feat
history_matches
pos
```

**Strengths**

- distinguishes recently displaced starters from permanent bench players;
- still compact.

**Weaknesses**

- slightly more correlated inputs.

### Option C — Expanded context

Option B plus:

```text
team_rot3
fdr
played_last
long_gap14
```

**Strengths**

- may capture tactical substitution context.

**Weaknesses**

- more noise;
- weak justification for fixture difficulty in cameo probability.

**Final decision:** Option B — Bench + role history.

```python
CAMEO_FEATURES = [
    "min_lag1",
    "min_ewm_hl2",
    "start_rate_hl3",
    "bench_streak",
    "days_feat",
    "history_matches",
    "pos",
]
```

---

# 10. Cameo-Minutes Feature Set

**Decision:** What should predict:

\[
E[M \mid Cameo, NotStart]
\]

?

### Option A — Same features as cameo probability

**Strengths**

- simplest implementation.

**Weaknesses**

- probability of appearing and duration after appearing are different processes.

### Option B — Duration-focused subset **(Recommended default)**

```text
min_lag1
min_ewm_hl2
bench_streak
days_feat
history_matches
pos
```

Later add historical cameo-duration features.

**Strengths**

- better conceptual match;
- avoids unnecessary role-classification features.

**Weaknesses**

- may initially omit useful context.

### Option C — Rich substitution-context model

Include:

- duration history;
- team substitution tendencies;
- congestion;
- opponent context.

**Strengths**

- highest eventual ceiling.

**Weaknesses**

- not suitable for the cleanest V2.0 baseline.

**Final decision:** Option B — Duration-focused subset.

```python
CAMEO_MIN_FEATURES = [
    "min_lag1",
    "min_ewm_hl2",
    "bench_streak",
    "days_feat",
    "history_matches",
    "pos",
]
```

---

# 11. P60 Architecture

**Decision:** How should the production system estimate:

\[
P(M \geq 60)
\]

?

### Option A — Direct binary classifier **(Recommended default)**

Train:

\[
P60=P(M\ge60 \mid X)
\]

directly.

**Strengths**

- already aligned with FPL scoring;
- already exists in V1;
- simple production contract.

**Weaknesses**

- may be inconsistent with start/cameo state probabilities.

### Option B — Structural P60

Estimate:

\[
P60 = P(Start)\times P(M\ge60\mid Start)
\]

**Strengths**

- probabilistically coherent;
- naturally links start and duration.

**Weaknesses**

- requires an additional conditional P60 head;
- errors compound.

### Option C — Direct + structural ensemble

Train both and blend them.

**Strengths**

- may combine complementary information.

**Weaknesses**

- adds another blending decision;
- should not be introduced without strong validation.

**Final decision:** Option A — Direct binary classifier. Production `P60` is the calibrated direct prediction `P(minutes >= 60 | X)`. Structural P60 is diagnostic only in V2.0.

---

# 12. P60 Feature Set

**Decision:** What should the direct P60 head use?

### Option A — Same features as start head **(Recommended default)**

**Strengths**

- simple;
- captures both selection and role stability.

**Weaknesses**

- may omit conditional substitution tendencies.

### Option B — Start features + duration features

Add:

- starter-duration history;
- recent P60 rate;
- recent substitution tendency.

**Strengths**

- more directly models the 60-minute boundary.

**Weaknesses**

- requires new engineered features.

### Option C — Minimal role-only P60

Use:

```text
min_lag1
min_ewm_hl2
start_lag1
start_rate_hl3
pos
```

**Strengths**

- extremely simple baseline.

**Weaknesses**

- may underfit the threshold event.

**Final decision:** Option A — Use the same baseline feature set as the start head for direct P60.

---

# 13. Expected-Minutes Composition Rule

**Decision:** What should be the primary final expected-minutes formula?

### Option A — Pure soft mixture **(Recommended default)**

\[
\boxed{
E[M]
=
p_s\mu_s+(1-p_s)p_c\mu_c
}
\]

**Strengths**

- law-of-total-expectation consistent;
- smooth;
- interpretable;
- no arbitrary threshold discontinuities.

**Weaknesses**

- depends heavily on probability calibration.

### Option B — Soft mixture + bounded correction model

Fit a second-stage model to correct systematic mixture residuals.

**Strengths**

- can learn stable residual bias.

**Weaknesses**

- risks recreating complexity;
- needs strict out-of-fold training.

### Option C — Hybrid threshold routing

Retain V1-style branch routing.

**Strengths**

- may reduce error in specific regimes.

**Weaknesses**

- discontinuous;
- hard to interpret;
- current 0.70–0.80 failure makes it suspect.

**Final decision:** Option A — Pure soft mixture.

```python
pred_exp_minutes = (
    p_start * pred_minutes_if_start
    + (1.0 - p_start) * p_cameo * pred_minutes_if_cameo
)
```

---

# 14. Starter Taper

**Decision:** Should `mu_start` be reduced when `P(start)` is low?

### Option A — Remove taper **(Recommended default)**

Use the raw conditional starter-duration prediction.

**Strengths**

- keeps conditional and state probability roles separate;
- avoids double-counting uncertainty.

**Weaknesses**

- may miss a genuine relationship between uncertain starts and early substitution.

### Option B — Learned taper

Add `p_start` as an input to the starter-duration model.

**Strengths**

- lets the model learn the relationship rather than imposing it.

**Weaknesses**

- requires strictly out-of-fold `p_start` during training.

### Option C — Keep manual taper

Retain current taper thresholds/functions.

**Strengths**

- easiest migration.

**Weaknesses**

- heuristic;
- difficult to justify;
- can distort expected values.

**Final decision:** Option A — Remove manual starter taper from V2.0. If a relationship later proves real, test a learned out-of-fold `p_start` feature instead.

---

# 15. Low-P(start) Minute Caps

**Decision:** Should final minutes be manually capped when `P(start)` is low?

### Option A — Remove modelling caps **(Recommended default)**

Only enforce physical bounds:

\[
0\le E[M]\le90
\]

**Strengths**

- probabilistically clean;
- exposes upstream calibration problems instead of masking them.

**Weaknesses**

- extreme predictions may occasionally appear during early development.

### Option B — Production-only warning caps

Do not modify the value, but flag predictions crossing suspicious combinations.

Example:

```text
p_start < 0.10
expected_minutes > 30
```

**Strengths**

- maintains model integrity;
- supports monitoring.

**Weaknesses**

- downstream product still receives unusual values unless explicitly handled.

### Option C — Retain hard caps

Keep current low-P(start) maximum-minute rules.

**Strengths**

- prevents visibly implausible outputs.

**Weaknesses**

- hides model inconsistency;
- adds non-probabilistic rules.

**Final decision:** Option A — Remove probability-based minute caps from modelling logic. Enforce only physical bounds and warnings.

---

# 16. Cameo Duration Bounds

**Decision:** What range should `mu_cameo` use?

### Option A — `[1, 59]`

**Strengths**

- reflects the idea that a cameo normally starts after kickoff.

**Weaknesses**

- not physically correct; an early injury substitution can produce >59 minutes.

### Option B — `[1, 90]` **(Recommended default)**

Allow any valid league-minute duration.

**Strengths**

- physically correct;
- avoids invalid assumptions.

**Weaknesses**

- rare extreme predictions may require monitoring.

### Option C — Data-derived percentile cap

Example:

\[
[1,Q_{99.5}(cameo\ minutes)]
\]

calculated on training data.

**Strengths**

- robust to extreme model predictions.

**Weaknesses**

- introduces another data-dependent threshold;
- must be recalculated chronologically.

**Final decision:** Option B — Use `[1, 90]` for cameo-duration support. Do not enforce a 59-minute cap.

---

# 17. Missing-Value Strategy

**Decision:** How should missing historical features be handled?

### Option A — Fill all missing with zero

**Strengths**

- matches much of the current implementation;
- simple.

**Weaknesses**

- treats unknown history as actual zero history;
- poor for cold starts.

### Option B — Native LightGBM missing values + evidence counts **(Recommended default)**

Keep NaNs and add fields such as:

```text
history_matches
season_history_matches
cold_start
```

**Strengths**

- preserves the meaning of unknown information;
- improves new-player handling.

**Weaknesses**

- requires careful schema validation.

### Option C — Training-set median imputation + missing flags

**Strengths**

- deterministic;
- useful for models that cannot natively handle NaN.

**Weaknesses**

- less natural for LightGBM;
- adds imputation artifacts.

**Final decision:** Option B — Preserve native LightGBM missing values and add `history_matches`, `season_history_matches`, and `cold_start`.

---

# 18. `played_last`

**Decision:** Should `played_last` remain?

### Option A — Remove from baseline **(Recommended default)**

Because:

\[
played\_last = I(min\_lag1 > 0)
\]

**Strengths**

- reduces redundancy;
- `min_lag1` contains richer information.

**Weaknesses**

- trees may benefit from the explicit binary split.

### Option B — Keep in baseline

**Strengths**

- cheap and interpretable;
- may help tree splitting.

**Weaknesses**

- redundant.

### Option C — Keep only if ablation improves performance

Start without it and add only after A11.

**Strengths**

- evidence-driven.

**Weaknesses**

- one more experiment.

**Final decision:** Option A — Remove `played_last` from the V2.0 baseline. Reintroduce only if ablation shows stable value.

---

# 19. `long_gap14`

**Decision:** Should `long_gap14` remain?

### Option A — Remove from baseline **(Recommended default)**

Use continuous `days_feat`.

**Strengths**

- avoids arbitrary 14-day discontinuity;
- less redundant.

**Weaknesses**

- may remove a useful nonlinear absence threshold.

### Option B — Keep both

**Strengths**

- lets trees use both continuous and thresholded rest.

**Weaknesses**

- arbitrary threshold;
- redundant.

### Option C — Replace with learned/categorical gap bands

Example:

```text
0–4
5–8
9–14
15+
```

**Strengths**

- richer nonlinear structure.

**Weaknesses**

- more feature engineering;
- should be evidence-driven.

**Final decision:** Option A — Remove `long_gap14` from the V2.0 baseline and retain continuous `days_feat`.

---

# 20. FDR

**Decision:** Should Fixture Difficulty Rating enter V2.0 minutes?

### Option A — Remove from V2.0 baseline **(Recommended default)**

Test later through ablation.

**Strengths**

- cleaner role model;
- FDR has a much stronger causal connection to scoring than minutes.

**Weaknesses**

- may miss fixture-specific rotation effects.

### Option B — Keep FDR

**Strengths**

- preserves current context;
- some managers rotate by opponent strength.

**Weaknesses**

- possible weak/noisy signal.

### Option C — Replace FDR later with richer opponent/context variables

Use opponent strength only when modelling tactical selection explicitly.

**Strengths**

- better modelling logic.

**Weaknesses**

- more future work.

**Final decision:** Option A — Remove FDR from the V2.0 baseline. Test later as an isolated ablation.

---

# 21. `team_rot3`

**Decision:** Should team-wide XI churn remain in V2.0?

### Option A — Remove from baseline

**Strengths**

- simplifies model;
- same signal currently applies equally to every player.

**Weaknesses**

- loses useful manager-rotation context.

### Option B — Keep initially, ablate later **(Recommended default)**

**Strengths**

- existing leak-free feature;
- plausible role-context signal;
- low implementation cost.

**Weaknesses**

- may be weaker than player-specific rotation.

### Option C — Remove and replace immediately with player-specific rotation

**Strengths**

- more targeted.

**Weaknesses**

- introduces new feature engineering during architectural migration.

**Final decision:** Option A — Exclude `team_rot3` from the canonical V2.0 feature lists. Test it later only as an isolated ablation after the core V2.0 system is established.

---

# 22. Calibration Candidate Set

**Decision:** Which calibrators should be considered?

### Option A — Raw vs isotonic

**Strengths**

- closest to current system.

**Weaknesses**

- isotonic can overfit sparse groups.

### Option B — Raw vs Platt vs isotonic **(Recommended default)**

**Strengths**

- tests low-variance and flexible calibration;
- lets data select method.

**Weaknesses**

- more artifacts and comparison logic.

### Option C — Platt only

**Strengths**

- simple;
- stable.

**Weaknesses**

- may underfit nonlinear calibration errors.

**Final decision:** Option B — Compare raw, Platt/logistic, and isotonic calibration for `P(start)`, `P(cameo | bench)`, and P60.

---

# 23. Calibration Selection Rule

**Decision:** How should a calibrator be selected?

### Option A — Lowest calibration-fold Brier score

**Strengths**

- straightforward.

**Weaknesses**

- can overfit one calibration period.

### Option B — Best mean chronological Brier skill with stability requirement **(Recommended default)**

Select calibrator only if it:

- improves mean Brier Skill Score;
- is not materially worse on any major fold;
- has sufficient calibration sample size.

Otherwise use raw probabilities.

**Strengths**

- robust;
- avoids calibration for calibration's sake.

**Weaknesses**

- slightly more implementation logic.

### Option C — Always isotonic when enough rows exist

**Strengths**

- deterministic.

**Weaknesses**

- insufficiently evidence-driven.

**Final decision:** Option B — Select calibration by mean chronological Brier Skill Score with a fold-stability requirement. If calibrated probabilities do not improve reliably, use raw probabilities.

---

# 24. Calibration Pooling

**Decision:** Should calibrators be position-specific?

### Option A — One global calibrator

**Strengths**

- maximum sample.

**Weaknesses**

- may ignore meaningful GK/outfield differences.

### Option B — GK + outfield calibrators with global fallback **(Recommended default)**

**Strengths**

- matches proposed model pooling;
- reasonable sample sizes.

**Weaknesses**

- extra artifacts.

### Option C — Four position calibrators

**Strengths**

- maximum specialization.

**Weaknesses**

- high overfitting risk, especially for cameo events.

**Final decision:** Option B — Use GK and pooled-outfield calibrators with hierarchical fallback.

A subgroup calibrator is eligible only when its calibration block contains:

- at least **200 rows**;
- at least **30 positive events**; and
- at least **30 negative events**.

Fallback order:

1. GK/outfield subgroup calibrator if thresholds are met;
2. global calibrator if the same thresholds are met globally;
3. raw model probability if no eligible calibrator exists.

These are V2.0 engineering safety thresholds and may be revisited only through a separately documented calibration ablation.

---

# 25. Chronological Validation Structure

**Decision:** What validation strategy should control model selection?

### Option A — One latest-season tail split

**Strengths**

- closest to current workflow;
- simple.

**Weaknesses**

- unstable;
- one season can dominate conclusions.

### Option B — Expanding walk-forward folds across seasons **(Recommended default)**

Example:

```text
Fold 1:
Train → seasons up to A
Calibration → next block
Evaluation → following block

Fold 2:
Expand train
Calibration → next block
Evaluation → following block
```

**Strengths**

- realistic forecasting;
- tests stability across time.

**Weaknesses**

- more compute.

### Option C — Rolling fixed-window folds

Train only the most recent N seasons before each evaluation block.

**Strengths**

- adapts to football evolution.

**Weaknesses**

- discards older information;
- adds another window-size decision.

**Final decision:** Option B — Use expanding walk-forward chronological folds across seasons with a fixed generation rule:

- Minimum training history: all available completed prior seasons plus all current-season GWs before the calibration block.
- Calibration block: 6 GWs.
- Evaluation block: 6 GWs.
- Step size: 6 GWs.
- Final holdout: the latest available chronological 6-GW block, never used for model, hyperparameter, feature, calibrator, or threshold selection.
- Every ablation must use the exact same centrally stored fold definitions.

---

# 26. Calibration Block Size

**Decision:** How much data should be reserved for probability calibration inside each fold?

### Option A — Fixed 15% chronological tail

**Strengths**

- matches current validation-style logic.

**Weaknesses**

- percentage can map to very different numbers of GWs.

### Option B — Fixed number of GWs **(Recommended default)**

Example:

```text
Calibration = final 5–8 GWs before evaluation block
```

Exact number must be locked before final testing.

**Strengths**

- interpretable;
- consistent football time horizon.

**Weaknesses**

- event counts vary by season.

### Option C — Minimum-event adaptive block

Expand until each head has enough positive examples.

**Strengths**

- statistically sensible for sparse events.

**Weaknesses**

- folds become less uniform;
- more implementation complexity.

**Final decision:** Option B — Use exactly the final 6 GWs immediately preceding each 6-GW evaluation block as the calibration block. Do not resize the block after seeing evaluation performance; sparse calibration cases use the fallback rules below.

---

# 27. Baseline Set

**Decision:** Which baselines are mandatory for every experiment?

### Option A — Minimal baselines

- previous-match minutes;
- EWMA minutes;
- V1.

**Strengths**

- cheap.

**Weaknesses**

- limited diagnosis.

### Option B — Full diagnostic baseline set **(Recommended default)**

Require:

- position mean;
- previous-match minutes;
- leak-free EWMA;
- `90 * P(start)`;
- uncalibrated soft mixture;
- V1 current production;
- calibrated V2 mixture.

**Strengths**

- clearly identifies where gains originate.

**Weaknesses**

- more reporting.

### Option C — Only V1 vs V2

**Strengths**

- simplest.

**Weaknesses**

- cannot tell whether either model beats simple heuristics.

**Final decision:** Option B — Require position mean, previous-match minutes, leak-free EWMA, `90 * P(start)`, uncalibrated soft mixture, current V1, and calibrated V2.

---

# 28. Primary Minutes Metric

**Decision:** What metric should determine expected-minutes model selection?

### Option A — MAE only

**Strengths**

- intuitive.

**Weaknesses**

- ignores severe misses and downstream threshold effects.

### Option B — MAE primary + RMSE and bias guardrails **(Recommended default)**

Primary:

\[
MAE
\]

Guardrails:

- RMSE;
- mean error;
- subgroup stability;
- downstream metrics.

**Strengths**

- robust and interpretable.

**Weaknesses**

- requires multi-metric decision logic.

### Option C — Downstream xPoints metric primary

**Strengths**

- directly tied to product goal.

**Weaknesses**

- hard to isolate minutes quality;
- downstream model noise can mask minutes improvements.

**Final decision:** Option B — Use MAE as primary expected-minutes metric, with RMSE, mean bias, subgroup stability, calibration, and downstream xPoints as guardrails.

---

# 29. Probability Metrics

**Decision:** What metrics are mandatory for `P(start)`, cameo, and P60?

### Option A — AUC + Brier

**Strengths**

- current-style minimal set.

**Weaknesses**

- incomplete calibration picture.

### Option B — Full probability diagnostics **(Recommended default)**

Require:

- AUC;
- Brier;
- prevalence Brier;
- Brier Skill Score;
- log loss;
- calibration intercept;
- calibration slope;
- reliability bins;
- sample counts.

**Strengths**

- separates ranking from calibration.

**Weaknesses**

- more reporting code.

### Option C — Brier only

**Strengths**

- directly evaluates probability quality.

**Weaknesses**

- loses discrimination diagnostics.

**Final decision:** Option B — Require AUC, Brier, prevalence Brier, Brier Skill Score, log loss, calibration intercept/slope, reliability bins, and sample counts.

---

# 30. State-Tree Metric

**Decision:** Should the entire Start/Cameo/DNP state distribution be scored?

### Option A — No state-level metric

Evaluate heads independently.

**Strengths**

- simpler.

**Weaknesses**

- misses errors in the combined state distribution.

### Option B — Multiclass log loss **(Recommended default)**

States:

```text
START
CAMEO
DNP
```

with probabilities:

\[
P(Start)=p_s
\]

\[
P(Cameo)=(1-p_s)p_c
\]

\[
P(DNP)=(1-p_s)(1-p_c)
\]

**Strengths**

- directly evaluates the state tree;
- highly relevant to current probability bottleneck.

**Weaknesses**

- adds another evaluation metric.

### Option C — Multiclass Brier score

**Strengths**

- probability-focused;
- interpretable extension of binary Brier.

**Weaknesses**

- less commonly reported.

**Final decision:** Option B — Use multiclass log loss for the three-state distribution: START, CAMEO, DNP.

---

# 31. Starter-Taper Diagnostic

**Decision:** How should the system determine whether start probability contains conditional duration information?

### Option A — Do not test

Assume taper is unnecessary.

**Strengths**

- simplest.

**Weaknesses**

- risks discarding real signal.

### Option B — Conditional duration by `P(start)` bins **(Recommended default)**

For actual starters, report:

```text
p_start band
N
mean actual starter minutes
starter-head prediction
MAE
```

**Strengths**

- directly answers whether low-confidence starters play fewer minutes.

**Weaknesses**

- diagnostic only; not itself a model.

### Option C — Include `p_start_oof` directly in starter-duration model

**Strengths**

- lets the model learn the effect.

**Weaknesses**

- requires careful out-of-fold probability generation.

**Final decision:** Option B — Report conditional starter duration by `P(start)` bins with N, mean actual starter minutes, starter-head prediction, and MAE.

---

# 32. Probability-Band Diagnostics

**Decision:** How should probability bands be reported?

### Option A — Fixed 0.10 bins

Example:

```text
0.0–0.1
...
0.9–1.0
```

**Strengths**

- intuitive.

**Weaknesses**

- sparse bins possible.

### Option B — Fixed bins + sample counts **(Recommended default)**

Always report:

- N;
- predicted mean;
- actual rate;
- minutes MAE.

**Strengths**

- prevents misleading small-sample conclusions.

**Weaknesses**

- none material.

### Option C — Equal-frequency deciles

**Strengths**

- similar N per bin.

**Weaknesses**

- less intuitive probability boundaries.

**Final decision:** Option B — Use fixed probability bins and always report N, mean predicted probability, actual rate, and minutes MAE.

---

# 33. Minimum Improvement Threshold

**Decision:** How much improvement is required to justify added complexity?

### Option A — Any positive mean improvement

**Strengths**

- maximizes raw metric optimization.

**Weaknesses**

- easily keeps noise.

### Option B — Predeclared practical threshold + consistency **(Recommended default)**

Keep added complexity only if it:

- improves mean walk-forward MAE by a meaningful threshold, **or**
- materially improves P60/state calibration/downstream xPoints;
- and the gain is directionally consistent across folds.

Suggested starting MAE threshold:

\[
\Delta MAE \ge 0.25 \text{ minutes}
\]

subject to confidence intervals.

**Strengths**

- prevents complexity creep.

**Weaknesses**

- threshold must be finalized.

### Option C — Statistical significance only

**Strengths**

- formal.

**Weaknesses**

- tiny practically meaningless improvements can be significant with enough rows.

**Final decision:** Option B — Keep added complexity only if it improves mean walk-forward MAE by at least 0.25 minutes or gives a clearly meaningful probability/downstream gain, with directionally consistent folds.

---

# 34. Early-Season Handling in V2.0

**Decision:** What should happen before season priors are implemented?

### Option A — Reset season history normally

**Strengths**

- simplest.

**Weaknesses**

- weak GW1–5 forecasts.

### Option B — Keep V2.0 simple but report cold-start diagnostics **(Recommended default)**

Use existing history policy for V2.0, but emit:

```text
cold_start
history_matches
season_history_matches
```

and evaluate early GWs separately.

Season priors become V2.1.

**Strengths**

- preserves clean architectural experiment;
- makes weakness measurable.

**Weaknesses**

- early-season predictions remain imperfect initially.

### Option C — Implement season priors immediately

**Strengths**

- stronger early-season production model.

**Weaknesses**

- mixes two major redesigns.

**Final decision:** Option B — Keep V2.0 architecturally clean. Emit `cold_start`, `history_matches`, and `season_history_matches`; add season-start priors in V2.1.

---

# 35. Season-Prior Strategy for V2.1

**Decision:** When priors are added, what form should they take?

### Option A — Fixed previous-season carryover

Use previous-season final EWM directly.

**Strengths**

- simple.

**Weaknesses**

- too confident after transfers/role changes.

### Option B — Shrinkage prior **(Recommended default)**

\[
r_t
=
\frac{\kappa r_{prior}+n_t r_{current}}
{\kappa+n_t}
\]

with prior downweighting for:

- transfer;
- manager change;
- promotion;
- long injury;
- position change.

**Strengths**

- principled;
- evidence-weighted.

**Weaknesses**

- requires tuning `kappa`.

### Option C — Separate ML cold-start model

Train a model specifically for GW1–5 role initialization.

**Strengths**

- flexible.

**Weaknesses**

- substantially more complex.

**Final decision:** Option B — In V2.1 use shrinkage priors `r_t = (kappa*r_prior + n_t*r_current)/(kappa+n_t)` with downweighting for transfers, manager changes, promotion, long injury, and position changes. Tune `kappa` chronologically.

---

# 36. Availability Context Timing

**Decision:** When should injury/suspension/status features enter the roadmap?

### Option A — V2.0

**Strengths**

- potentially large immediate gain.

**Weaknesses**

- high timestamp-leakage risk.

### Option B — V2.2 after clean V2 baseline **(Recommended default)**

First prove the architecture, then add timestamp-safe availability data.

**Strengths**

- preserves attribution;
- lower implementation risk.

**Weaknesses**

- leaves some obvious context unused initially.

### Option C — Do not include external availability

**Strengths**

- fully statistical/reproducible from match data.

**Weaknesses**

- puts a hard ceiling on minutes forecasting quality.

**Final decision:** Option B — Add timestamp-safe availability/status features in V2.2 after V2.0 is stable.

---

# 37. Congestion and Cup/Europe Context Timing

**Decision:** When should non-league schedule context be introduced?

### Option A — V2.0

**Strengths**

- directly relevant to rotation.

**Weaknesses**

- expands data pipeline immediately.

### Option B — V2.2/V2.3 **(Recommended default)**

Add only after V2.0 and season-prior experiments are stable.

**Strengths**

- clean staged development.

**Weaknesses**

- congestion effects remain unmodelled at first.

### Option C — Never model explicitly

Let recent role/history absorb rotation.

**Strengths**

- simpler.

**Weaknesses**

- may systematically miss known rotation pressure.

**Final decision:** Option B — Add non-league congestion and cup/Europe context in V2.2/V2.3 after the baseline and prior stages are stable.

---

# 38. Player-Specific Rotation Timing

**Decision:** When should player-level rotation tendency replace/supplement `team_rot3`?

### Option A — Immediately

**Strengths**

- more targeted than team rotation.

**Weaknesses**

- new feature work during migration.

### Option B — V2.3 after baseline stability **(Recommended default)**

Test features such as:

\[
P(Start_t \mid Start_{t-1})
\]

and congestion-conditioned start retention.

**Strengths**

- staged and testable.

**Weaknesses**

- delays possible gain.

### Option C — Keep team-level rotation only

**Strengths**

- simple.

**Weaknesses**

- same risk signal for every player.

**Final decision:** Option B — Add player-specific rotation features in V2.3 and compare them against `team_rot3`.

---

# 39. Final Expected-Minutes Output Name

**Decision:** What should the canonical field be called?

### Option A — `pred_minutes`

**Strengths**

- compatible with existing downstream code.

**Weaknesses**

- doesn't explicitly communicate expectation.

### Option B — `pred_exp_minutes` **(Recommended default)**

**Strengths**

- clearly communicates expected value;
- reduces ambiguity.

**Weaknesses**

- may require downstream renaming/migration.

### Option C — Write both aliases

```text
pred_minutes
pred_exp_minutes
```

with identical values.

**Strengths**

- easiest transition.

**Weaknesses**

- duplicate schema fields can create future inconsistency.

**Final decision:** Option B — Use `pred_exp_minutes` as the canonical V2 output field.

---

# 40. Required V2.0 Output Schema

**Decision:** How much component detail should production files expose?

### Option A — Minimal

```text
pred_exp_minutes
p_start
p60
```

**Strengths**

- compact.

**Weaknesses**

- difficult to diagnose.

### Option B — Full state decomposition **(Recommended default)**

Require:

```text
p_start_raw
p_start_cal
pred_minutes_if_start
p_cameo_raw
p_cameo_cal
pred_minutes_if_cameo
p_start_state
p_cameo_state
p_dnp_state
p60_raw
p60_cal
pred_exp_minutes
history_matches
season_history_matches
cold_start
model_version
fallback_flags
```

**Strengths**

- auditable;
- supports downstream uncertainty;
- makes debugging straightforward.

**Weaknesses**

- wider files.

### Option C — Production minimal + separate audit artifact

Keep output small and write a second detailed diagnostic file.

**Strengths**

- cleaner production contract.

**Weaknesses**

- more file coordination.

**Final decision:** Option B — Emit the full state decomposition.

Required fields:

```text
p_start_raw
p_start_cal
pred_minutes_if_start
p_cameo_raw
p_cameo_cal
pred_minutes_if_cameo
p_start_state
p_cameo_state
p_dnp_state
p60_raw
p60_cal
pred_exp_minutes
history_matches
season_history_matches
cold_start
model_version
fallback_flags
```

---

# 41. Production Guardrail Policy

**Decision:** What should happen when a prediction looks implausible?

### Option A — Silently clamp to heuristic values

**Strengths**

- clean-looking outputs.

**Weaknesses**

- hides failures.

### Option B — Physical clipping + warnings only **(Recommended default)**

Enforce only:

\[
0\le p\le1
\]

\[
1\le\mu_{start}\le90
\]

\[
1\le\mu_{cameo}\le90
\]

\[
0\le E[M]\le90
\]

Flag unusual combinations.

**Strengths**

- preserves model meaning;
- makes failures observable.

**Weaknesses**

- unusual but valid outputs may reach downstream systems.

### Option C — Separate production correction layer

Keep raw model prediction plus corrected production value.

**Strengths**

- auditable compromise.

**Weaknesses**

- reintroduces dual prediction logic.

**Final decision:** Option B — Use physical clipping plus warnings only. Probabilities `[0,1]`, conditional durations `[1,90]`, final expected minutes `[0,90]`.

---

# 42. Confirmed Absence / Suspension Overrides

**Decision:** How should hard external information be applied?

### Option A — Modify model features only

Feed status into the model and let it decide.

**Strengths**

- single prediction system.

**Weaknesses**

- confirmed absence should mathematically imply zero.

### Option B — Separate timestamped override layer **(Recommended default)**

Preserve raw model prediction, then apply an explicit production override such as:

```text
confirmed unavailable -> expected_minutes = 0
```

Store:

- raw prediction;
- final prediction;
- reason;
- source;
- information timestamp.

**Strengths**

- correct;
- auditable;
- avoids contaminating statistical logic.

**Weaknesses**

- requires override infrastructure.

### Option C — Ignore external overrides

**Strengths**

- pure model.

**Weaknesses**

- knowingly produces bad forecasts.

**Final decision:** Option B — Apply confirmed absences/suspensions through a separate timestamped override layer preserving the raw model prediction and provenance.

---

# 43. Versioning Strategy

**Decision:** How should V2 artifacts be versioned?

### Option A — Continue simple `vN`

Example:

```text
v1
v2
v3
```

**Strengths**

- simple.

**Weaknesses**

- doesn't distinguish architecture generations.

### Option B — Semantic architecture versions **(Recommended default)**

Example:

```text
minutes/v2.0/
minutes/v2.1/
minutes/v2.2/
```

with run metadata inside.

**Strengths**

- directly reflects roadmap;
- easier reproducibility.

**Weaknesses**

- may require path changes.

### Option C — Hash/config-based run IDs

**Strengths**

- strongest reproducibility.

**Weaknesses**

- less human-friendly.

**Final decision:** Option B — Use semantic architecture versions such as `minutes/v2.0`, `minutes/v2.1`, and `minutes/v2.2`, with run metadata underneath.

---

# 44. Reproducibility Metadata

**Decision:** What must every trained model save?

### Option A — Model files + metrics only

**Strengths**

- minimal.

**Weaknesses**

- insufficient for exact reproduction.

### Option B — Full model card metadata **(Recommended default)**

Save:

- code version;
- feature version;
- seasons;
- chronological folds;
- training dates;
- feature lists;
- hyperparameters;
- random seeds;
- calibration method;
- sample counts;
- positive rates;
- data paths/hashes where possible;
- metrics;
- creation timestamp.

**Strengths**

- production-grade reproducibility.

**Weaknesses**

- more metadata plumbing.

### Option C — Full dataset snapshots

Save exact training datasets with every model.

**Strengths**

- strongest possible reproducibility.

**Weaknesses**

- storage-heavy;
- data duplication.

**Final decision:** Option B — Save full model-card metadata: code version, feature version, seasons, folds, training dates, exact feature lists, hyperparameters, seeds, calibration method, sample counts, positive rates, data identifiers/hashes where possible, metrics, and creation timestamp.

---

# 45. V1 Migration Policy

**Decision:** How should V1 be handled during V2 rollout?

### Option A — Replace immediately after offline tests

**Strengths**

- simple.

**Weaknesses**

- higher operational risk.

### Option B — Shadow V1 and V2 in parallel **(Recommended default)**

For several live GWs:

- produce both forecasts;
- compare calibration;
- compare downstream xPoints;
- monitor data failures;
- retain rollback.

**Strengths**

- safest migration;
- validates real-time pipeline behaviour.

**Weaknesses**

- temporary duplicate compute/storage.

### Option C — Keep both permanently as ensemble candidates

**Strengths**

- preserves model diversity.

**Weaknesses**

- unnecessary complexity unless ensemble proves useful.

**Final decision:** Option B — Run V1 and V2 in parallel shadow mode for several live GWs before cutover and preserve immediate version-based rollback.

---

# 46. V2.0 Acceptance Rule

**Decision:** What must happen before V2 becomes production?

### Option A — Lower expected-minutes MAE than V1

**Strengths**

- simple.

**Weaknesses**

- ignores calibration/downstream value.

### Option B — Multi-gate acceptance **(Recommended default)**

Require:

1. lower mean chronological expected-minutes MAE than V1 and strongest simple baseline;
2. positive `P(start)` Brier Skill Score;
3. no unexplained severe probability-band discontinuity;
4. P60 calibration at least non-inferior to V1;
5. state-tree probability quality improved or non-inferior;
6. no material subgroup collapse;
7. downstream xPoints improved or non-inferior;
8. reproducible artifacts and rollback path.

**Strengths**

- aligns model quality with actual system usage.

**Weaknesses**

- more demanding.

### Option C — Downstream xPoints improvement only

**Strengths**

- directly product-focused.

**Weaknesses**

- can hide a poorly specified minutes model.

**Final decision:** Option B — Use multi-gate production acceptance with bootstrap-defined non-inferiority.

Require:

1. lower mean chronological expected-minutes MAE than V1 and the strongest simple baseline;
2. positive aggregate `P(start)` Brier Skill Score;
3. no severe unexplained probability-band discontinuity;
4. P60 and state-probability performance must be non-inferior to V1;
5. no material subgroup collapse;
6. downstream xPoints performance must be non-inferior to V1;
7. full reproducibility and rollback readiness.

**Non-inferiority rule:** use player- or fixture-block bootstrap confidence intervals on the paired V2−V1 metric difference. For loss metrics where lower is better (for example Brier score, log loss, MAE), V2 is non-inferior only when the **upper bound of the 95% bootstrap confidence interval** does not exceed the predeclared practical deterioration margin.

For V2.0, set the practical margins to:

- P60 Brier score: **+0.002** maximum allowed deterioration;
- state-tree log loss: **+0.005** maximum allowed deterioration;
- downstream xPoints MAE: **+0.01** maximum allowed deterioration.

The bootstrap must preserve player/fixture dependence through block resampling rather than independent row resampling.

---

# 47. Practical MAE Improvement Threshold

**Decision:** What minimum MAE gain should justify a more complex variant?

### Option A — `0.10` minute

**Strengths**

- sensitive.

**Weaknesses**

- likely within noise.

### Option B — `0.25` minute + fold consistency **(Recommended default)**

A component should improve mean walk-forward MAE by at least:

\[
0.25
\]

minutes **or** provide a clearly meaningful probability/downstream gain.

**Strengths**

- practical;
- discourages complexity creep.

**Weaknesses**

- should be revisited after bootstrap variability is measured.

### Option C — `0.50` minute

**Strengths**

- very conservative.

**Weaknesses**

- may reject useful smaller improvements.

**Final decision:** Option B — Require at least a 0.25-minute mean walk-forward MAE improvement to justify added complexity unless there is a clearly meaningful probability or downstream-xPoints gain; require fold consistency.

---

# 48. Uncertainty Requirement

**Decision:** Should V2.0 expose a player-role uncertainty measure?

### Option A — No additional uncertainty metric

Use `p_start`, `p60`, and expected minutes separately.

**Strengths**

- simplest.

**Weaknesses**

- no single role-risk indicator.

### Option B — State entropy **(Recommended default)**

Using:

\[
P(Start),P(Cameo),P(DNP)
\]

calculate:

\[
H=-\sum_i p_i\log p_i
\]

**Strengths**

- directly reflects state uncertainty;
- cheap;
- useful for player-page reliability logic.

**Weaknesses**

- not directly measured in minutes.

### Option C — Predictive minutes variance/distribution

**Strengths**

- richer uncertainty.

**Weaknesses**

- requires distributional modelling beyond current V2 scope.

**Final decision:** Option B — Expose state entropy `H = -sum(p_i * log(p_i))` from START/CAMEO/DNP as the V2.0 role-uncertainty metric.

---

# 49. V2.1+ Deferred Roadmap

**Decision:** How should deferred improvements be ordered?

### Option A — Priors → availability → congestion → player rotation **(Recommended default)**

Sequence:

```text
V2.1 season-start priors
V2.2 availability/status
V2.3 congestion + player rotation
V2.4 distributional uncertainty
```

**Strengths**

- addresses largest known structural gaps in a logical order.

**Weaknesses**

- later interaction effects may require retesting earlier components.

### Option B — Availability first

```text
V2.1 availability
V2.2 priors
V2.3 congestion
V2.4 player rotation
```

**Strengths**

- availability may provide the largest immediate predictive gain.

**Weaknesses**

- requires timestamped data engineering sooner.

### Option C — Priors + availability together

**Strengths**

- faster path to richer model.

**Weaknesses**

- weak attribution.

**Final decision:** Option A — Roadmap: V2.1 season-start priors → V2.2 availability/status → V2.3 congestion + player-specific rotation → V2.4 distributional uncertainty.

---

# 50. Final Implementation Lock

Complete this section only after all decisions above are filled in.

## Required V2.0 architecture

```text
Position pooling:
GK separate + pooled outfield

Position encoding:
Native categorical `pos`

Start training population:
Fixture-eligible players

Fixture eligibility:
Roster member + no confirmed pre-deadline suspension, transfer-away, or definite unavailability

Start features:
`min_lag1`, `min_ewm_hl2`, `start_lag1`, `start_rate_hl3`, `start_streak`, `bench_streak`, `days_feat`, `history_matches`, `pos`

Starter-duration features:
`min_lag1`, `min_ewm_hl2`, `start_rate_hl3`, `days_feat`, `history_matches`, `pos`

Cameo target definition:
`P(cameo | available and not started)`

Cameo-probability features:
`min_lag1`, `min_ewm_hl2`, `start_rate_hl3`, `bench_streak`, `days_feat`, `history_matches`, `pos`

Cameo-duration features:
`min_lag1`, `min_ewm_hl2`, `bench_streak`, `days_feat`, `history_matches`, `pos`

P60 approach:
Direct binary classifier; structural P60 diagnostic only

P60 features:
Same baseline feature set as start head

Expected-minutes composition:
Pure soft mixture: `p_start*mu_start + (1-p_start)*p_cameo*mu_cameo`

Starter taper:
Removed from V2.0

Low-P(start) caps:
Removed; physical bounds + warnings only

Cameo duration bounds:
`[1, 90]`

Missing-value policy:
Native NaNs + `history_matches`, `season_history_matches`, `cold_start`

played_last:
Removed from V2.0 baseline; ablation candidate

long_gap14:
Removed from V2.0 baseline; ablation candidate

FDR:
Removed from V2.0 baseline; later ablation

team_rot3:
Excluded from canonical V2.0; isolated later ablation only

Calibration candidates:
Raw, Platt, isotonic

Calibration selection:
Best mean chronological Brier Skill Score with fold-stability requirement; otherwise raw

Calibration pooling:
GK + outfield if N≥200 with ≥30 positive and ≥30 negative events; otherwise global under same thresholds; otherwise raw probability

Chronological fold strategy:
Expanding walk-forward; training = all prior completed seasons + current pre-calibration history; calibration = 6 GWs; evaluation = 6 GWs; step = 6 GWs; latest 6-GW block reserved as untouched final holdout

Calibration block:
Final 6 GWs before each evaluation block

Mandatory baselines:
Position mean; previous match; EWMA; `90*P(start)`; uncalibrated soft mixture; V1; calibrated V2

Primary minutes metric:
MAE primary; RMSE, bias, subgroup stability, calibration and downstream xPoints as guardrails

Probability metrics:
AUC; Brier; prevalence Brier; BSS; log loss; calibration intercept/slope; reliability bins; counts

State-tree metric:
Three-state multiclass log loss

Minimum complexity improvement:
≥0.25 min mean walk-forward MAE or meaningful probability/xPoints gain, with fold consistency

Early-season V2.0 policy:
No new season prior; expose cold-start/history diagnostics; priors deferred to V2.1

Production output field:
`pred_exp_minutes`

Output schema:
Full state-decomposition schema

Guardrail policy:
Physical clipping + warnings only

External override policy:
Separate timestamped override layer preserving raw model prediction

Versioning:
`minutes/v2.0`, `minutes/v2.1`, ... plus run metadata

Reproducibility metadata:
Full model-card metadata

Migration strategy:
V1/V2 live shadow period before cutover

Production acceptance rule:
Multi-gate acceptance with paired block-bootstrap non-inferiority: 95% CI upper bound must remain within +0.002 P60 Brier, +0.005 state log loss, and +0.01 xPoints MAE deterioration margins

Uncertainty output:
Three-state entropy

Deferred roadmap:
V2.1 priors → V2.2 availability → V2.3 congestion/player rotation → V2.4 distributional uncertainty
```

---

# Final Hardening Locks

The following previously unresolved decisions are now fixed:

1. **`team_rot3`:** excluded from canonical V2.0; later isolated ablation only.
2. **Walk-forward protocol:** 6-GW calibration + 6-GW evaluation, advancing by 6 GWs, with the latest 6-GW block held out completely from model selection.
3. **Calibration evidence:** subgroup/global calibrators require `N >= 200`, at least 30 positive events, and at least 30 negative events; otherwise fall back hierarchically to raw probabilities.
4. **Non-inferiority:** paired player/fixture block bootstrap, 95% CI, with practical deterioration margins of `+0.002` P60 Brier, `+0.005` state-tree log loss, and `+0.01` downstream xPoints MAE.

With these locks, implementation should not introduce new modelling rules outside the contract. Any change should be treated as a versioned experiment or ablation.

# Implementation-Ready Definition

The V2 design becomes **hardened and ready for implementation** when:

- every `Final decision` field is completed;
- Section 50 contains no unresolved entries;
- feature lists are exact rather than descriptive;
- training populations are unambiguous;
- chronological folds are reproducible;
- calibration selection is deterministic;
- output fields have fixed names;
- guardrails do not silently alter statistical predictions;
- V1 remains reproducible as the migration benchmark;
- V2.1+ features are clearly excluded from V2.0 unless explicitly selected.

Once these conditions are satisfied, implementation should move from model-design discussion to:

1. code architecture;
2. unit tests;
3. prediction-level instrumentation;
4. ablation runner;
5. chronological backtesting;
6. V1-versus-V2 shadow comparison.

