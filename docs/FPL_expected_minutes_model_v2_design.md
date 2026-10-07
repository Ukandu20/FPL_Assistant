# FPL Expected-Minutes Model V2: Suggested System Design

> Status: historical proposal. Superseded for implementation by the
> [hardened V2 contract](FPL_expected_minutes_V2_implementation_contract_HARDENED.md).
> Use [V2 operations](FPL_EXPECTED_MINUTES_V2.md) for current commands.
> References to missing local source files and ?latest? metrics below describe
> the original review, not the current checkout.

## 1. Purpose

This document proposes a simpler, probabilistically coherent replacement for the current expected-minutes system in `minutes_model_builder.py`. The goal is not to add more modelling layers. It is to preserve the useful conditional heads, remove post-processing that cannot justify itself out of sample, improve probability calibration, and make every extra component measurable through chronological ablation tests.

The model must support two separate downstream needs:

1. a well-calibrated expected-minutes value for expected-points calculations; and
2. explicit probabilities such as `P(start)` and `P(minutes >= 60)`, because FPL scoring is nonlinear around appearances and the 60-minute boundary.

The proposed V2 should be considered successful only if it beats simple baselines on repeated chronological backtests, improves downstream expected-points accuracy, and remains stable across seasons and probability bands.

---

## 2. Evidence from the latest run

The latest reported `metrics.json` findings are:

| Metric | Latest value | Interpretation |
|---|---:|---|
| Test observations | 1,825 | Large enough for an initial diagnosis, not enough to settle all subgroup choices |
| Overall expected-minutes MAE | 28.67 | Too high for a core expected-points input |
| Final MAE on actual starters | 31.45 | The final prediction is poor on this realised-state subset |
| Raw starter-minutes head MAE on actual starters | 11.60 | The conditional duration head is promising |
| Final MAE on actual non-starters | 21.97 | The final prediction is also weak on this realised-state subset |
| Raw bench/cameo head MAE | 13.47 | The conditional bench branch is promising |
| `P(start)` AUC | 0.7791 | Useful ranking ability, but not strong enough to ignore calibration |
| `P(start)` Brier score | 0.2097 | Approximately no better than the inferred constant-rate baseline of about 0.207 |
| `P(60+)` AUC | 0.7666 | Useful but moderate discrimination |
| `P(60+)` Brier score | 0.2027 | Must be compared with its own prevalence baseline |
| Starter MAE in the `P(start) = 0.70-0.80` band | 60.79 | Severe local failure or unstable routing; requires sample count and code-path audit |

### Important interpretation constraint

The 11.60 and 13.47 conditional-head MAEs are **not directly comparable** with the 28.67 unconditional expected-minutes MAE. The conditional scores use the realised future state—whether the player actually started or was benched—to select the relevant head. At forecast time that state is unknown.

Therefore, the metrics do not prove that mixing alone creates the full error gap. They do show that:

- the conditional duration regressors contain useful signal;
- state-probability quality is a major bottleneck;
- hard routing, tapering, caps, or calibration may be amplifying the error; and
- the discontinuity around the 0.70-0.80 start-probability band needs direct investigation.

The correct comparison is current final prediction versus a **soft mixture built from exactly the same out-of-fold head predictions**.

---

## 3. Current design: retained concepts and challenged concepts

The current implementation appears to contain five prediction heads:

1. `P(start)`;
2. `E[minutes | start]`;
3. `P(cameo | not start)`;
4. `E[minutes | cameo, not start]`; and
5. `P(minutes >= 60)`.

It also uses some combination of global and position-specific models, isotonic calibration, start-probability routing thresholds, goalkeeper-specific routing, starter-minute tapering, low-`P(start)` caps, and position-specific bench caps.

### Keep by default

- The separation between start probability and conditional starter duration.
- The two-part cameo branch: probability of appearing and duration if used.
- Leak-free lagged and exponentially weighted features.
- Chronological validation and early stopping.
- A direct `P(60+)` prediction because of the FPL scoring discontinuity.
- Separate uncertainty outputs rather than only one predicted-minutes number.

### Remove from the V2 baseline

- Hard routing between bench, mixture, and starter predictions.
- Manual starter-duration tapering based on `P(start)`.
- Core modelling dependence on low-$P(start)$ minute caps.
- Automatic use of separate DEF, MID, and FWD heads.
- Automatic use of isotonic calibration for every subgroup.

These components may return only if they produce repeatable improvements on untouched chronological folds. A production guardrail is not evidence that the underlying forecast is correct.

---

## 4. Proposed architecture

### 4.1 Core state tree

For fixture $t$, define:

- $S_t = 1$ if the player starts;
- $C_t = 1$ if the player appears after not starting;
- $M_t$ as recorded league minutes, bounded to $[0, 90]$ for normal FPL fixtures.

The model estimates:

$$
p_s = P(S_t = 1 \mid X_t)
$$

$$
\mu_s = E[M_t \mid S_t = 1, X_t]
$$

$$
p_c = P(C_t = 1 \mid S_t = 0, X_t)
$$

$$
\mu_c = E[M_t \mid C_t = 1, S_t = 0, X_t]
$$

The unconditional expected minutes are then:

$$
\hat{M}_t = p_s\mu_s + (1-p_s)p_c\mu_c
$$

This is the primary production prediction. No threshold selects one branch. A player with $p_s = 0.79$ should not receive a fundamentally different calculation from one with $p_s = 0.80$.

The complete implied state distribution is:

$$
P(\text{start}) = p_s
$$

$$
P(\text{bench cameo}) = (1-p_s)p_c
$$

$$
P(0\text{ minutes}) = (1-p_s)(1-p_c)
$$

These probabilities must sum to one within numerical tolerance.

### 4.2 Direct 60-minute head

Retain a direct classifier:

$$
p_{60}^{direct} = P(M_t \geq 60 \mid X_t)
$$

Expected minutes do not uniquely determine $P(60+)$. Two players can have the same mean but very different distributions. The direct head should therefore remain available for appearance points, clean-sheet entitlement, and risk reporting.

Also compute a structural consistency estimate:

$$
p_{60}^{struct} = p_s \times P(M_t \geq 60 \mid S_t=1, X_t)
$$

because a non-starting cameo rarely reaches 60 league minutes. The direct and structural estimates should be compared during evaluation. Do not average them without out-of-sample evidence.

### 4.3 Model pooling strategy

Begin with:

- one goalkeeper model family; and
- one pooled outfield model family with position as a categorical feature.

This is a stronger default than four independent position families because it preserves sample size while still allowing nonlinear position interactions. Compare it against:

1. one global model with position;
2. goalkeeper plus pooled outfield; and
3. four independent position families.

Choose the smallest architecture whose gain is consistent across seasons, not merely the architecture with the best single split.

---

## 5. Feature design

Every feature must be available at the prediction timestamp and must be calculated with a strict shift so fixture $t$ cannot influence its own predictors.

### 5.1 Core role-history features

| Feature | Definition | Decision |
|---|---|---|
| $min_lag1$ | Minutes in the immediately previous eligible fixture | Keep: fast role-change signal |
| $min_ewm_hl2$ | Shifted EWMA of recent minutes, half-life about two matches | Keep: smoothed role baseline |
| $start_lag1$ | Whether the player started the previous fixture | Keep |
| $start_rate_hl3$ | Shifted EWMA of starts, half-life about three matches | Keep |
| $start_streak$ | Consecutive starts before the target fixture | Keep initially; ablate |
| $bench_streak$ | Consecutive non-starts before the target fixture | Keep initially; ablate |
| $played_last$ | Whether $min_lag1 > 0$ | Ablate because it is derived from $min_lag1$ |

For a lagged series $z$, an EWMA can be written as:

$$
EWM_t = \alpha z_{t-1} + (1-\alpha)EWM_{t-1}
$$

where the half-life $h$ implies:

$$
\alpha = 1 - 2^{-1/h}
$$

$min_lag1$ and $min_ewm_hl2$ should both remain in the baseline. They are correlated but conceptually different: the first captures the latest shock; the second captures the recent role.

### 5.2 Availability and evidence-strength features

Add, when reliably timestamped:

- official availability status before the deadline;
- chance-of-playing or injury-confidence field;
- suspension flag and remaining suspension fixtures;
- return-from-injury flag;
- transfer/new-club flag;
- promoted-team/new-to-league flag;
- number of historical league observations;
- days since last appearance and days since last start;
- explicit missing-history indicators.

Do not encode missing history as genuine zero history. LightGBM can handle missing values directly. Add evidence-strength fields such as $history_matches$ so the model can learn when lags are unreliable.

Availability data has unusually high value for minutes, but it is also vulnerable to timestamp leakage. A status published after the FPL deadline must never enter the corresponding training row.

### 5.3 Schedule and rotation context

Candidate features are:

- days since the club's previous match in any competition;
- days until the club's next match in any competition;
- matches in the surrounding 7-, 14-, and 21-day windows;
- European/cup match proximity;
- international-break return and travel proxy;
- recent manager change;
- team XI churn from previous fixtures only;
- player-specific rotation rate under congestion;
- player-specific start retention: $P(start_t | start_{t-1})$;
- player-specific bench recovery: $P(start_t | not start_{t-1})$.

$team_rot3$ is a reasonable team-level context feature, but it gives the same risk signal to every player. A player-specific rotation tendency is more targeted and should be tested as a replacement or interaction.

### 5.4 Fixture and identity context

- position as a categorical feature, not an arbitrary ordinal scale where possible;
- home/away only if it improves backtests;
- opponent/FDR only as an ablation candidate;
- club identity or manager identity only with careful regularisation and season-forward testing.

FDR is plausible for tactical rotation but has a weaker causal link to minutes than to attacking output. It stays only if it improves start Brier score or final minutes metrics consistently.

### 5.5 Feature groups by head

| Head | Primary feature groups |
|---|---|
| $P(start)$ | Role history, availability, schedule congestion, player/team rotation, season prior, position |
| $\text{E[M \| start]}$ | Recent conditional minutes, substitution pattern, congestion, availability/return status, position |
| $P(\text{cameo} \| \text{bench})$ | Bench history, prior cameo rate, availability, bench streak, position, match/schedule context |
| $E[M \| \text{cameo, bench}]$ | Prior cameo durations, substitution timing tendency, position, limited fixture context |
| $P(60+)$ | Start-role features plus conditional duration history and availability |

Do not assume every head needs the same feature list. The relevance of fixture context to coming on may differ from its relevance to cameo duration.

---

## 6. Targets and training populations

| Head | Target | Training rows |
|---|---|---|
| Start classifier | $1[started]$ | All eligible player-fixture rows |
| Starter duration | Observed minutes | Rows where $started = 1$ |
| Cameo classifier | $1[minutes > 0]$ | Rows where $started = 0$ |
| Cameo duration | Observed minutes | Rows where $started = 0$ and $minutes > 0$ |
| P60 classifier | $1[minutes >= 60]$ | All eligible player-fixture rows |

Eligibility rules must be explicit. In particular, distinguish:

- registered squad players who could plausibly appear;
- unavailable players known before the deadline;
- players missing only because of incomplete source data; and
- players not at the club for the target fixture.

If the production candidate set includes all registered players but training contains only players appearing in match logs, the start and zero-minute targets will be selection-biased.

---

## 7. Training logic

### 7.1 Chronological folds

Use expanding-window or rolling-origin validation. A suitable pattern is:

- train on all information before cutoff $c$;
- fit calibration only on a later calibration block;
- evaluate once on the following untouched block;
- repeat over several cutoffs and seasons.

Never use random row splits. Rows for the same player and adjacent fixtures are highly dependent, and random splitting overstates generalisation.

### 7.2 Out-of-fold composition

Every evaluation of the final mixture must use out-of-fold predictions from **all** component heads. The sequence for each fold is:

1. fit each head on the fold's training period;
2. predict raw probabilities and durations on its calibration period;
3. fit candidate calibrators using only that calibration period;
4. freeze models and calibrators;
5. predict the untouched evaluation period;
6. compose expected minutes from those frozen predictions; and
7. calculate all metrics and subgroup diagnostics.

This prevents optimistic evaluation caused by mixing in-sample head predictions.

### 7.3 Losses and constraints

- Use binary log loss for classifiers during training, then select models using Brier score, log loss, calibration, and downstream utility—not AUC alone.
- Use MAE, Huber, or another robust regression objective for conditional durations; compare rather than assume.
- Clip conditional starter duration to the feasible support, normally $[1, 90]$.
- Clip cameo duration to $[1, 59]$ unless the data definition provides a valid counterexample.
- Clip final expected minutes only to $[0, 90]$ as a numerical safety check.

Clipping should almost never activate. Log activation rates and investigate if it does.

### 7.4 Hyperparameter selection

Tune each head using the same chronological folds used for model selection. Prefer small search spaces and conservative trees because subgroup heads can have limited samples. Record:

- training and validation date ranges;
- row counts and positive rates;
- feature schema and data versions;
- model parameters and random seeds;
- calibration method and sample size; and
- fold-level, not only pooled, results.

---

## 8. Probability calibration

### 8.1 Candidate calibrators

Compare for $P(start)$, $P(cameo | bench)$, and $P(60+)$:

1. no calibration;
2. Platt/logistic scaling; and
3. isotonic regression.

Isotonic calibration is flexible but can overfit small position or cameo samples. It should not be the automatic choice.

### 8.2 Selection criteria

Select calibration separately for each probability head using untouched chronological folds and:

- Brier score;
- Brier skill score;
- log loss;
- expected calibration error with fixed, documented bins;
- reliability curves;
- calibration slope and intercept; and
- stability across seasons and positions.

The Brier skill score is:

$$
BSS = 1 - \frac{BS_{model}}{BS_{reference}}
$$

where the reference is a training-period prevalence forecast or another forecast available at that time. A negative BSS means the model is worse than the reference.

### 8.3 Sparse-group fallback

Use hierarchical fallback rules:

- subgroup calibrator if it exceeds a predeclared sample and positive-event threshold;
- otherwise pooled goalkeeper/outfield calibrator;
- otherwise global calibrator;
- otherwise raw probability with an explicit warning flag.

The threshold must be chosen before inspecting the final test block.

---

## 9. Inference logic

For each player-fixture row:

1. build timestamp-valid features;
2. attach season-start priors and evidence-strength fields;
3. select goalkeeper or outfield model family;
4. predict raw $p_start$, $\mu_start$, $p_cameo$, $\mu_cameo$, and $p60$;
5. apply the frozen calibrators to probability heads;
6. calculate the soft mixture;
7. run invariant and input-quality checks;
8. emit prediction, component probabilities, uncertainty flags, and model metadata.

Reference composition:

```python
p_start = calibrate_start(start_model.predict_proba(X_start))
mu_start = clip(start_minutes_model.predict(X_start), 1.0, 90.0)

p_cameo = calibrate_cameo(cameo_model.predict_proba(X_cameo))
mu_cameo = clip(cameo_minutes_model.predict(X_cameo), 1.0, 59.0)

p_zero = (1.0 - p_start) * (1.0 - p_cameo)
p_bench_cameo = (1.0 - p_start) * p_cameo

expected_minutes = (
    p_start * mu_start
    + p_bench_cameo * mu_cameo
)
expected_minutes = clip(expected_minutes, 0.0, 90.0)
```

Required output fields should include:

- $\text{expected\_minutes}$;
- calibrated and raw $p_\text{start}$;
- $\mu_\text{minutes\_if\_start}$;
- calibrated and raw $p_\text{cameo\_if\_bench}$;
- $\mu_\text{minutes\_if\_cameo}$;
- calibrated and raw $p_{60}$;
- $p_\text{zero\_minutes}$;
- $p_\text{bench\_cameo}$;
- history sample count;
- season-prior weight;
- model/fold/data version; and
- warning or fallback flags.

Do not overwrite a probabilistic prediction with a manually asserted lineup state unless the product explicitly supports a separate, timestamped expert-override layer. Preserve both the model output and override with provenance.

---

## 10. Season-start priors

Resetting all lagged features at the season boundary throws away useful role information. Carrying the previous season forward without shrinkage is also unsafe because squads, managers, fitness, and roles change.

### 10.1 Prior construction

For a role statistic $r$, define the initial prior:

$$
r_{prior} = w_{prev}r_{previous\ season} + (1-w_{prev})r_{peer}
$$

where $r_{peer}$ is a position/team or league-position prior available before GW1.

Update it as current-season evidence arrives:

$$
r_t = \frac{\kappa r_{prior} + n_t r_{current,t}}{\kappa + n_t}
$$

where:

- $n_t$ is the amount of current-season evidence;
- $\kappa$ is the effective prior sample size; and
- the prior weight is $\kappa / (\kappa + n_t)$.

Tune $\kappa$ chronologically. Do not hard-code “use the prior for five gameweeks” without testing; evidence should decay by observations, not only calendar week.

### 10.2 Prior adjustments

Downweight the previous-season component for:

- transfer to a new club;
- new manager or major tactical change;
- promoted team or arrival from another league;
- long injury absence;
- changed position;
- newly signed competition for the same role; and
- small previous-season sample.

For a player with no relevant history, use the peer prior and expose a $cold\_start$ flag. Never convert unknown history into a confident zero-role estimate.

### 10.3 Prior features

Candidate prior features include:

- previous-season start rate;
- previous-season minutes per squad-eligible fixture;
- previous-season minutes conditional on starting;
- previous-season cameo rate and duration;
- previous-season final-10-fixture EWM;
- prior sample size; and
- transfer, manager-change, promotion, and long-absence indicators.

Evaluate early-season performance separately for GW1-3, GW4-6, and GW7+.

---

## 11. Evaluation framework

### 11.1 Mandatory baselines

V2 must beat:

- $B0$: training-period mean minutes by position;
- $B1$: previous-match minutes;
- $B2$: leak-free EWMA minutes;
- $B3$: $90 * P(start)$;
- $B4$: the uncalibrated soft mixture;
- $B5$: the current production implementation; and
- $B6$: a simple season-prior/EWMA blend for early gameweeks.

### 11.2 Final expected-minutes metrics

Report:

- MAE;
- median absolute error;
- RMSE, to expose severe misses;
- mean error/bias;
- pinball loss at useful quantiles if interval models are added;
- error by position, season, gameweek band, availability state, and history depth;
- error by predicted $P(start)$ decile with sample counts;
- error by predicted-minutes band; and
- error in the 50-70 expected-minutes region.

Do not make starter-only MAE the primary objective for an unconditional forecast. It is a diagnostic because it conditions on the realised outcome.

### 11.3 Probability metrics

For start, cameo, and P60 probabilities, report:

- AUC as a ranking diagnostic;
- Brier score;
- prevalence Brier baseline;
- Brier skill score;
- log loss;
- calibration intercept and slope;
- expected calibration error;
- reliability tables/plots with counts; and
- precision-recall AUC for sparse events such as cameos if appropriate.

### 11.4 Distribution and consistency checks

Report:

- $p_start + p_bench_cameo + p_zero = 1$ error;
- direct versus structural P60 disagreement;
- percentage of clipped predictions;
- monotonicity/smoothness across adjacent $P(start)$ bins;
- prediction drift by season and team;
- missing-feature and fallback rates; and
- cold-start versus established-player performance.

### 11.5 Downstream FPL metrics

Minutes are an input, not the final product. Also evaluate:

- appearance-point MAE or log loss;
- 60-minute appearance-point calibration;
- clean-sheet entitlement calibration;
- downstream expected-points MAE and rank correlation;
- captaincy or transfer decision regret for a defined decision rule; and
- calibration of total FPL points where the minutes model materially contributes.

A small minutes-MAE gain that worsens P60 calibration may be a net loss to the actual FPL system.

### 11.6 Uncertainty

Use player- or fixture-block bootstrap confidence intervals and report fold-to-fold variation. Treat a change as a genuine improvement only when it is directionally consistent across seasons and its uncertainty is acceptably small.

---

## 12. Required ablation programme

Run all variants on identical frozen chronological folds and store prediction-level outputs.

| ID | Variant | Question answered |
|---|---|---|
| A0 | EWMA baseline | Does the ML system beat a strong simple role model? |
| A1 | Current production system | What is the exact migration reference? |
| A2 | Same heads + pure soft mixture | Does removing routing improve the final prediction? |
| A3 | A2 without taper | Was taper double-counting start uncertainty? |
| A4 | A3 without probability-based caps | Were caps masking upstream inconsistency? |
| A5 | Uncalibrated probabilities | Does calibration genuinely help? |
| A6 | Platt calibration | Does a low-variance calibrator beat raw/isotonic? |
| A7 | Isotonic calibration | Is extra flexibility justified? |
| A8 | Global model + position feature | Does maximum pooling work best? |
| A9 | GK + pooled outfield | Is goalkeeper separation the best trade-off? |
| A10 | Four position families | Do fully separate heads earn their complexity? |
| A11 | Remove `played_last` | Is it redundant with `min_lag1`? |
| A12 | Remove `long_gap14` | Is it redundant with continuous days-since features? |
| A13 | Remove FDR | Does fixture difficulty add stable minutes signal? |
| A14 | Remove `team_rot3` | Does team-level churn help? |
| A15 | Add player rotation tendency | Is player-specific rotation more useful? |
| A16 | Native missing values + evidence count | Does this improve cold starts over zero imputation? |
| A17 | Add season-start priors | Does early-season accuracy improve without later harm? |
| A18 | Add timestamped availability | What is the value of real-world status information? |
| A19 | Direct versus structural P60 | Which probability is better calibrated downstream? |

For each ablation, report the delta versus its immediate parent and versus A0/A1. Avoid changing multiple components in one comparison unless the experiment explicitly tests a bundle.

### First diagnostic experiment

Before retraining, reconstruct from the latest run's saved row-level predictions:

$$
\hat M_{soft}=p_s\mu_s+(1-p_s)p_c\mu_c
$$

Compare it with the current final prediction on the same 1,825 rows. Break the delta down by:

- `P(start)` decile, especially 0.70-0.80;
- current routing branch;
- position;
- actual start state;
- caps/taper activation; and
- sample count.

This is the cheapest test of the main diagnosis. If prediction-level components were not saved, add them before the next run.

---

## 13. Guardrails and invariants

Guardrails should detect failure, not silently redesign the forecast.

### 13.1 Hard invariants

- all probabilities are finite and within `[0, 1]`;
- `mu_start` is within `[1, 90]`;
- `mu_cameo` is within `[1, 59]` under the standard league-minute definition;
- expected minutes are within `[0, 90]`;
- state probabilities sum to one;
- no feature timestamp is later than the prediction cutoff;
- feature columns and ordering match the trained model schema;
- model and calibrator versions are compatible.

### 13.2 Data-quality fallbacks

If current input data are missing or stale:

1. use the documented prior/peer fallback;
2. reduce confidence and expose a flag;
3. preserve the missing value for the model where supported; and
4. log the reason and affected population.

Do not silently fill all missing values with zero.

### 13.3 Monitoring alerts

Alert on:

- calibration deterioration beyond a predeclared tolerance;
- negative Brier skill score over a meaningful rolling window;
- sharp error discontinuities between adjacent probability bins;
- unusual clipping, fallback, or missingness rates;
- team or position drift;
- abrupt feature-distribution shifts;
- season-start cold-start failures; and
- disagreement between direct and structural P60 beyond a validated range.

### 13.4 Optional external overrides

Known suspensions, confirmed absences, and confirmed lineups may justify overrides, but the override system must be separate from the statistical model. Store:

- original prediction;
- override value;
- reason and source;
- information timestamp;
- author/process; and
- downstream result.

This makes expert information auditable and prevents leakage into historical evaluation.

---

## 14. Migration plan

### Phase 0: Reproduce and instrument

- Freeze the current code, data snapshot, split, seed, and `metrics.json` as the reference run.
- Save row-level raw/calibrated head predictions, routing branch, taper amount, cap activations, and final output.
- Recalculate the reported metrics from the saved prediction table.
- Add counts to every subgroup and probability band.
- Verify the 0.70-0.80 failure is not a tiny-sample or metric-labelling issue.

**Exit criterion:** the 28.67 reference result is reproducible and every post-processing transformation is observable.

### Phase 1: Shadow soft mixture

- Implement the pure mixture alongside the current system without changing production outputs.
- Reuse the exact existing head predictions to isolate composition logic.
- Compare overall, subgroup, calibration, and downstream FPL metrics.

**Exit criterion:** a written decision on routing, taper, and caps supported by identical-row results.

### Phase 2: Calibration rebuild

- Produce reliability tables for start, cameo, and P60.
- Add prevalence baselines and Brier skill scores.
- Compare raw, Platt, and isotonic calibration on nested chronological blocks.
- Introduce sparse-group fallbacks.

**Exit criterion:** selected calibrators improve probability metrics consistently and do not worsen expected-minutes or downstream metrics.

### Phase 3: Simplify pooling and features

- Compare global, GK/outfield, and four-position model families.
- Run redundancy and questionable-feature ablations.
- Switch from blanket zero imputation to native missingness plus evidence-strength features.

**Exit criterion:** the smallest model within a predeclared performance tolerance is selected.

### Phase 4: Add high-value context

- Add season-start priors.
- Add timestamp-safe availability and congestion signals.
- Add player-specific rotation tendency.
- Test each group incrementally.

**Exit criterion:** improvements survive multiple season-forward folds, especially GW1-6 and high-uncertainty players.

### Phase 5: Production shadow and cutover

- Run current and V2 models in parallel for several live gameweeks.
- Monitor inputs, drift, calibration, guardrail activations, and downstream expected-points effects.
- Define rollback based on model version, not manual code edits.
- Cut over only after the acceptance gates below are met.

**Exit criterion:** V2 is stable in shadow mode and has an immediate rollback path.

### Phase 6: Remove legacy complexity

- Delete or archive routing, taper, and cap paths only after V2 cutover is stable.
- Preserve the legacy implementation and artefacts needed for reproducibility.
- Update model cards, data contracts, tests, and monitoring documentation.

---

## 15. Acceptance gates

Exact numerical thresholds should be set before evaluating the final holdout. At minimum, require:

1. lower overall expected-minutes MAE than both the current 28.67 run and the strongest simple baseline;
2. positive start Brier skill score on aggregate and no material seasonal collapse;
3. no unexplained adjacent-bin discontinuity like the reported 60.79 MAE in the 0.70-0.80 band;
4. P60 calibration that is at least as good as the current system;
5. no material degradation in early-season, goalkeeper, availability, or cold-start segments;
6. improved or non-inferior downstream expected-points performance;
7. low and explainable clipping/fallback rates;
8. consistent gains across multiple chronological folds; and
9. full reproducibility from versioned inputs and configuration.

A useful decision rule is to prefer the simpler model unless the more complex variant produces a meaningful, stable, confidence-supported gain.

---

## 16. Recommended implementation order

The highest-value sequence is:

1. save prediction-level components and reproduce the latest run;
2. calculate the pure soft mixture from the same predictions;
3. audit the 0.70-0.80 band and all routing/taper/cap activations;
4. repair probability calibration and add Brier skill reporting;
5. select global versus GK/outfield pooling;
6. remove features that fail ablation;
7. add native missingness and evidence-strength features;
8. introduce season-start priors; and
9. add timestamped availability, congestion, and player-specific rotation context.

Do not begin by tuning the starter or cameo duration regressors. Their reported conditional MAEs are already the strongest part of the latest run. The first priority is to prove that the system turns those components and calibrated state probabilities into a coherent unconditional forecast.

---

## 17. Final proposed contract

The production model should be explainable in one sentence:

> Expected minutes equal the calibrated chance of starting times expected minutes when starting, plus the calibrated chance of not starting but appearing times expected cameo minutes.

Everything beyond that core—position splits, calibrators, contextual features, priors, or overrides—must have a named experiment, an out-of-sample result, and a documented fallback. That constraint is not merely aesthetic: it is what makes future improvements attributable, testable, and safe.

---

## 18. Source and scope note

This design is based on the reviewed `minutes_model_builder.py` architecture and the latest `metrics.json` findings discussed in the “Evaluate Models In Depth” task. Those two files are not present in this project's local mirror at the time of writing, so implementation-specific names and reported values should be revalidated against the canonical files before code migration begins. Synced files under `sources/` were treated as read-only.
