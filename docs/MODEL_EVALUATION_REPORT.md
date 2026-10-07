# FPL Assistant Model Evaluation Report

> Status: historical evaluation dated 2026-08-09. Artifact paths, metrics, and
> ?current?/?latest? claims below describe that evaluation snapshot. They are
> not a fresh assessment or production approval. See [V2 operations](FPL_EXPECTED_MINUTES_V2.md)
> and the [documentation index](README.md) for maintained entry points.


> Architecture update (2026-09-10): the mirrored script implementations described
> in this historical review have been consolidated. The backtest harness is now
> `fpl_assistant.testing.backtest_harness`; optimizer/data/golden checks are in
> `tests/` and participate in normal pytest discovery. Model-quality findings
> below remain historical observations, not a new evaluation.
**Evaluation date:** 9 August 2026  
**Repository:** `FPL_Assistant`  
**Scope:** active predictive models, the expected-points layer, captain ranking, and squad/transfer optimization. Legacy or duplicate entry points are noted separately.

## Executive summary

The project has a sensible decomposed architecture: minutes, attacking returns, defense, and saves are estimated separately, then translated into FPL expected points and passed to decision models. This is interpretable and makes component-level diagnosis possible. The goals/assists model is currently the strongest statistical component: its saved 2024–25 evaluation has low calibration error, and its rolling 2025–26 forecasts retain good event-probability Brier scores. The minutes model is the most consequential weakness because its errors propagate into nearly every downstream component. The saved latest minutes version is also materially worse than its earlier `v1` artifact.

The most urgent findings are:

1. **Model selection and artifact integrity are unsafe.** `latest_version.txt` points to `v5` for goals/assists, but that version directory has medians/audits and no trained model files. The expected-points latest pointer selects `v3`, which has no attached actuals or evaluation metrics.
2. **Minutes performance regressed.** Saved holdout MAE worsened from 22.84 minutes (`v1`) to 28.67 (`v5`), start Brier worsened from 0.135 to 0.210, and rolling 2025–26 minutes MAE is 28.38.
3. **Defense submodels do not generalize uniformly.** Rolling team clean-sheet AUC is 0.532 and DCP AUC is 0.476 despite a historical player clean-sheet AUC of 0.704. The empty saved DCP metrics file hides this weakness.
4. **The saves model has no useful rolling ranking signal.** Across seven available 2025–26 gameweeks, saves MAE is 1.81 and correlation with actual saves is -0.016.
5. **The trained discipline model is effectively degenerate and is not used by the active points forecaster.** Its saved outputs predict zero yellow and red cards on average; `points_forecast.py` instead applies fixed position priors.
6. **The active xPoints total omits computed negative components.** `points_forecast.py` calculates `xp_concede_penalty` and `xp_discipline_prior`, but neither is included in the `comp_cols` summed into `xPts`. Rolling 2025–26 MAE is 1.95 points, but correlation with actual points is only 0.214. Low MAE alone is misleading because most player-game outcomes are near zero.
7. **The optimizer lacks essential validation.** Saved squads contain six or seven midfielders, violating the required 2/5/5/3 squad composition, and one run records a £1,000m budget. Its reported objective is therefore not evidence of decision quality.
8. **There is no automated model test suite in `tests/`.** A backtest harness and optimizer invariant scripts exist under `scripts/tests/`, but they are not part of the normal pytest suite and no saved rolling backtest report was found.

Overall assessment: **promising research pipeline, not yet reliable for unattended production recommendations**. The best next investment is a reproducible rolling-origin evaluation and model registry gate, followed by minutes-model repair and optimizer constraint tests.

## Evaluation method and limitations

This review used:

- implementation inspection in [`src/fpl_assistant/models`](../src/fpl_assistant/models) and [`src/fpl_assistant/optimizers`](../src/fpl_assistant/optimizers);
- saved holdout metrics and model artifacts under [`data/models`](../data/models);
- saved rolling forecasts under [`data/predictions`](../data/predictions);
- actual 2025–26 player-gameweek data in [`merged_gws.csv`](<../data/processed/fpl/ENG-Premier League/2025-2026/gws/merged_gws.csv>);
- the existing but apparently unexecuted [`backtest_harness.py`](../src/fpl_assistant/testing/backtest_harness.py).

For the rolling check, each `GWn_m.csv` file was treated as the forecast available for its first named gameweek `n`; this avoids counting overlapping windows repeatedly. Predictions were joined to actuals on gameweek and `player_id`. Coverage was 96.2% for minutes, defense, and points, 96.5% for goals/assists, and 98.4% for saves. The sample contains gameweeks 1–8, except saves, which has seven gameweeks. These are diagnostic results, not definitive season-long estimates. File timestamps/provenance do not independently prove that every saved forecast was generated before its deadline, so the results should be labelled **rolling-artifact checks**, not certified leak-free backtests.

The stored `expected_points/v1` and `v2` outputs appear to use different schemas and row universes. Their metrics are therefore reported only within-version and should not be compared as a clean version experiment.

## Model inventory

| Component | Type | Primary target/output | Downstream role | Current evidence |
|---|---|---|---|---|
| Minutes | LightGBM mixture/classification system | start, cameo, 60+, expected minutes | exposure for every scoring component | Holdout + rolling check |
| Goals/assists | Per-position LightGBM and Poisson/Tweedie heads + isotonic calibration | goals, assists, return probabilities | attacking xPoints | Holdout + rolling check |
| Defense | Team CS classifier, team GC regressor, per-position DCP rate models | CS, goals conceded, defensive contribution | CS, concession, DCP xPoints | Holdout + rolling check |
| Saves | GK LightGBM and Poisson/Tweedie rate heads | saves per 90 / per match | goalkeeper save points | Holdout + rolling check |
| Discipline | Four LightGBM rate heads, optional Poisson heads | YC, RC, own goals, missed penalties | negative points | No persisted metrics; output audit only |
| Expected points | Deterministic probabilistic aggregator | component and total xPoints | common decision signal | Historical artifact + rolling check |
| Captain ranker | LightGBM LambdaRank | top-five/captain ordering | captain recommendation | Six-GW holdout |
| Squad/transfer optimizers | MILP via PuLP/CBC | legal squad, XI, transfers, chips | final action plan | Saved solutions; no outcome backtest |

## Cross-model results

### Saved holdout metrics

| Model | Evaluation slice | Principal results | Interpretation |
|---|---|---|---|
| Minutes `v1` | 4,138 rows | MAE 22.84; start AUC 0.849; start Brier 0.135 | Best saved minutes version |
| Minutes `v5` / latest | 1,825 rows | MAE 28.67; start AUC 0.779; start Brier 0.210 | Clear regression, though test windows differ |
| Goals/assists `v1` | 2,982 rows, 2024–25 GW30+ | goal MAE 0.169; assist MAE 0.141 using Poisson; Brier 0.078/0.065; ECE 0.0127/0.0084 | Good calibration on a short tail holdout |
| Defense `v1` | 2024–25 evaluation tail | player CS AUC 0.704; Brier 0.124; team ECE 0.068; GC MAE 1.186 | Moderate CS discrimination; weak GC accuracy |
| Saves `v1` | 252 GK rows, 2024–25 GW30+ | match MAE 1.313 using mean head + predicted minutes | Small test set; Poisson head is slightly worse |
| Captain `v6` | Six GWs | NDCG@5 0.375; precision@5 0.133; hit@5 0.167 | Weak and highly uncertain ranking performance |

### Rolling 2025–26 artifact check

| Model | Rows / groups | Result |
|---|---:|---|
| Minutes | 3,223 player-GWs | MAE 28.38; RMSE 42.30; bias -2.78; correlation 0.401; P60 Brier 0.228 |
| Goals | 3,614 player-GWs | count MAE 0.155; Brier 0.0495; bias +0.055 |
| Assists | 3,614 player-GWs | count MAE 0.130; Brier 0.0497; bias +0.029 |
| Team clean sheet | 160 team-games | Brier 0.201; AUC 0.532 |
| Player clean sheet | 3,223 player-GWs | Brier 0.124; AUC 0.593 |
| Goals conceded | 160 team-games | MAE 1.072; bias +0.207 |
| DCP | 3,018 outfield player-GWs | Brier 0.383; AUC 0.476 |
| Saves | 181 GK-GWs | MAE 1.806; bias +0.293; correlation -0.016 |
| Total xPoints | 3,223 player-GWs | MAE 1.948; RMSE 2.911; bias -0.302; correlation 0.214 |

The attacking Brier scores are helped by class imbalance, so they must be compared against base-rate and simple historical-rate baselines in the next evaluation. Similarly, point MAE must be paired with ranking and decision metrics.

## 1. Minutes model

### Overview

[`minutes_model_builder.py`](../src/fpl_assistant/models/minutes_model_builder.py) is a mixture-of-experts system rather than a single regressor. It estimates:

- probability of starting, with global and position-specific LightGBM classifiers;
- expected minutes conditional on starting;
- probability of a cameo when benched;
- expected cameo minutes conditional on appearing;
- probability of reaching 60 minutes;
- a threshold-based route between starter and bench predictions.

Features include lagged minutes and starts, exponentially weighted minutes/start rates, days since last match, gap and streak features, position, fixture difficulty, and a leak-free three-match team-rotation feature. Optional isotonic calibration and position-specific bench caps are present.

### Detailed logic

1. **Build the observation table.** Player fixture-calendar rows from all requested seasons are normalized, ordered by player and match date, and clipped to 0–120 observed minutes. The last requested season is the test season.
2. **Create strictly lagged state.** For player (i) before fixture (t), the builder calculates previous minutes `min_lag1`, previous-played indicator, lagged minutes EWMA with half-life 2, previous-start indicator, lagged start-rate EWMA with half-life 3, and consecutive start/bench streaks. Days since the player's prior match is converted to either `log1p(days)` or a capped raw value; `long_gap14` flags gaps above 14 days.
3. **Estimate team rotation.** For every team fixture, the code builds an XI membership vector, measures the absolute change from the preceding XI, shifts this change by one fixture, averages the prior three changes, and divides by 11. Therefore `team_rot3` at (t) uses XI information only through (t-1).
4. **Attach context.** Position is encoded as GK=0, DEF=1, MID=2, FWD=3. Fixture difficulty is joined from the selected form table. Missing FDR and rotation currently become zero.
5. **Split chronologically.** Training contains earlier seasons plus rows before `first_test_gw` in the final season; testing contains the final season from that gameweek onward. Training rows without the two core lag features are dropped.
6. **Fit the starting gate.** Global and, where supported, position-specific LightGBM binary classifiers estimate
   \[
   p_s=P(\text{starts}\mid X).
   \]
   Position and global predictions are blended by `gate_blend`. Optional monotonic constraints force selected history features to move start probability in the expected direction, and optional isotonic models calibrate the raw probabilities.
7. **Fit conditional minutes.** A LightGBM L2 regressor trained only on starters estimates \(\mu_s=E[M\mid\text{start},X]\). A separate direct classifier estimates \(p_{60}=P(M\ge60\mid X)\). Starter minutes can be tapered downward when (p_s) is uncertain.
8. **Fit the bench branch.** Global and position-specific classifiers estimate \(p_c=P(M>0\mid\text{benched},X)\). Position-specific L1 regressors trained on bench appearances estimate \(\mu_c=E[M\mid\text{benched and appears},X]\). The unconditional bench estimate is
   \[
   \hat M_b=p_c\mu_c,
   \]
   capped by either learned position-specific 95th-percentile cameo caps or a global cap. If a position has no bench DNPs in training, its cameo probability is forced to one.
9. **Route the final estimate.** The soft mixture is \(\hat M_{mix}=p_s\hat M_s+(1-p_s)\hat M_b\). Above `t_hi`, the starter head is used; below `t_lo`, the bench head is used; between thresholds, outfield players use the mixture. With `no_mix_gk`, goalkeepers use the starter head throughout the middle band. Low-start-probability caps can restrict the final estimate to 10 or 30 minutes before clipping to 0–120.
10. **Derive FPL exposure.** Appearance probability is
    \[
    p_{play}=p_s+(1-p_s)p_c,
    \]
    and expected appearance points are \(p_{play}+p_{60}\). The output retains all gate and conditional-head values for downstream use and diagnosis.

### Strengths

- The decomposition matches football selection mechanics and is much more defensible than direct unconditional minutes regression.
- Chronological splitting, lagging, and the prior-XI rotation calculation show explicit anti-leak design.
- Position-specific heads, global fallbacks, missing-class handling, calibration, tapering, and caps provide operational resilience.
- Metrics distinguish started and benched players and include probability metrics, not only aggregate MAE.
- The output exposes component heads, making poor routing diagnosable.

### Weaknesses

- Latest saved performance is poor and worse than `v1`: MAE increased 25.5%, while start Brier increased 55.0%. Different test sizes mean this is not a controlled comparison, which is itself a model-governance problem.
- The mixture's final MAE is much worse than the conditional raw heads (`v5`: 31.45 versus 11.60 for starters; 21.97 versus 13.47 for bench), pointing to gate/routing error rather than weak conditional regressors.
- The 0.7–0.8 start-probability band has catastrophic MAE of 60.79 minutes. That suggests unstable calibration, a selection-regime mixture, or threshold behavior near this band.
- Rolling P60 Brier of 0.228 is weak; because P60 directly drives appearance and clean-sheet points, this error is amplified downstream.
- Fixed thresholds/caps and one tail split are tuned and evaluated too closely. There is no nested rolling validation or uncertainty interval.
- Missing values are filled with zero for most minute features. Zero is semantically different from unknown for form, gap, and rotation variables.
- No explicit injury, suspension, predicted lineup, manager-change, congestion, European-match, or transfer/new-signing signal is visible in the core feature set.

### Improvements

1. Make `v1` the temporary champion and block promotion of `v5` until it beats the champion on identical rolling folds.
2. Replace a single tail split with weekly rolling-origin folds across at least two seasons; report MAE by position, starter state, price band, team, and prediction band.
3. Diagnose the 0.7–0.8 gate band with calibration curves and confusion matrices. Prefer a continuous mixture `p_start × E[min|start] + (1-p_start) × p_cameo × E[min|cameo]` over hard routing unless rolling validation proves thresholds help.
4. Add squad-status features available before deadline: injury/news status, chance of playing, suspension, predicted lineup consensus, recent starts in all competitions, days/rest, schedule congestion, manager tenure, and same-position competition.
5. Calibrate start, cameo, and P60 probabilities out of fold, not on a tail reused for other tuning decisions.
6. Publish quantiles or a discrete minutes distribution. The optimizer needs uncertainty around 0/1–59/60–90 outcomes, not only a mean.
7. Add baseline comparisons: last minutes, last-three mean, start-rate × 75 minutes, and official FPL chance-of-playing rules.

## 2. Goals and assists model

### Overview

[`goals_assists_model_builder.py`](../src/fpl_assistant/models/goals_assists_model_builder.py) trains global and position-specific per-90 models for goals and assists. It combines nonlinear LightGBM mean heads with optional log-link Poisson/Tweedie heads, scales by predicted minutes, converts rates to event probabilities, and calibrates probabilities by position using isotonic regression. Its feature set includes venue/FDR, availability, previous minutes, team attack and opponent defense strength, rolling actual and expected goal/assist rates, and exponentially weighted shot/SOT rates.

### Detailed logic

1. **Load played-match features.** The builder reads `players_form.csv` and optional team-form data for every selected season. Goals (`gls`) and assists (`ast`) are numeric targets; missing goalkeeper labels are replaced with zero. Goalkeepers may then be excluded entirely with `--skip-gk`.
2. **Construct context and form.** `venue_bin` and venue-specific FDR are combined with days since last match, active status, lagged minutes, team attacking strength, and opponent defensive strength. Available rolling goal, non-penalty xG, assist, xAG, shot, and shot-on-target rates survive only if their non-null coverage meets `na_thresh`. Shot/SOT EWMAs are shifted before calculation and may use position-specific half-lives.
3. **Define per-90 targets.** On played rows, observed goals and assists are divided by `minutes/90`. The chronological split uses all earlier seasons and the pre-cutoff part of the last season for training; the final-season tail is held out.
4. **Fit four model families.** Separate goal and assist LightGBM regressors are fitted globally and by position when a position has at least 150 rows. Optional Tweedie regressors with `power=1` and log link provide Poisson-like count-rate heads, again global and by position. LightGBM handles missing values internally; the Tweedie heads use training-set median imputation.
5. **Resolve sparse positions.** At prediction time, a position-specific head is used if fitted; otherwise the global head supplies the rate. Every predicted rate is clipped at zero.
6. **Attach exposure.** Test rows are joined to the minutes artifact on season, GW, date, and player. With `require_pred_minutes`, any missing join is fatal. Per-match count means are
   \[
   \hat g=\hat g_{90}\frac{\hat M}{90},\qquad
   \hat a=\hat a_{90}\frac{\hat M}{90}.
   \]
7. **Convert rates to event probabilities.** The Poisson/Tweedie rate is preferred when available. If the full minutes mixture is present, the goal probability is
   \[
   p_g=p_s(1-e^{-\lambda_{g,s}})+(1-p_s)p_c(1-e^{-\lambda_{g,c}}),
   \]
   where \(\lambda_{g,s}=g_{90}M_s/90\) and \(\lambda_{g,c}=g_{90}M_c/90\). Assists use the same construction. Without mixture fields, the fallback is \(1-e^{-r_{90}\hat M/90}\).
8. **Calibrate by position.** When enabled, the last 15% of training rows forms a chronological calibration tail. Isotonic regressions map raw goal and assist probabilities to observed occurrence separately by position. The final any-return probability assumes goal/assist independence:
   \[
   p_{return}=1-(1-p_g)(1-p_a).
   \]
9. **Persist diagnostics.** Predictions include both rate families, per-match means, probabilities, and true labels. Feature lists, medians, importances, missingness deltas, row-level missing-feature audits, reliability tables, model files, metadata, and metrics are written to latest and versioned locations.

### Strengths

- The per-90/exposure separation is statistically and operationally appropriate.
- Expected-stat features (`npxG`, `xAG`, shots, SOT) are stronger leading indicators than raw FPL returns alone.
- Position-specific models reduce structural heterogeneity, with a global fallback for sparse cases.
- Poisson heads outperform mean heads materially on the saved holdout: goal MAE improves from 0.226 to 0.169 and assist MAE from 0.177 to 0.141.
- Calibration is explicitly measured with Brier and ECE and backed by reliability artifacts.
- Training medians and missingness audits improve inference reproducibility.
- Rolling attacking probabilities remain the strongest component-level results.

### Weaknesses

- The holdout covers only GW30 onward in one test season; rare-event calibration can look stable over a small and unusually scheduled tail.
- Brier scores are not compared with prevalence-only, bookmaker, or FPL `xP` baselines. Without skill scores, absolute values overstate evidence.
- Isotonic calibration per position can overfit sparse return events unless generated from out-of-fold predictions.
- Goals and assists are modeled separately, while their dependence, team scoring environment, and shared minutes uncertainty are not propagated.
- The feature set lacks explicit penalty/set-piece share, role changes, teammate availability, likely formation, and player/team finishing priors.
- Point forecasts use selected probabilities/means without clear version-locked provenance; the current `latest_version.txt` points to incomplete `v5` artifacts.
- Rolling forecasts slightly overpredict both goals and assists, suggesting a mild calibration shift.

### Improvements

1. Repair the version registry immediately: a version cannot become latest until all expected models, metadata, feature schema, tests, and metrics exist.
2. Evaluate rolling log loss, Brier skill score, calibration intercept/slope, top-k return recall, and rank correlation against historical-rate and bookmaker baselines.
3. Use cross-fitted calibration, with shrinkage toward a global calibrator where position samples are small.
4. Add penalty/set-piece responsibility, projected team goal total, starting probability, role/position changes, and teammate-absence features.
5. Model goal/assist counts jointly or simulate them through a team-goal allocation model to preserve plausible dependence and ceiling outcomes.
6. Propagate the full minutes distribution rather than multiplying by a point estimate.
7. Track drift in shot-quality inputs and retrain/calibrate when rolling thresholds fail.

## 3. Defense model

### Overview

[`defense_model_builder.py`](../src/fpl_assistant/models/defense_model_builder.py) contains three related heads:

- a team-match LightGBM clean-sheet classifier, optionally isotonic-calibrated;
- a team goals-conceded regressor;
- position-specific defensive-contribution per-90 regressors, converted to threshold probabilities with a Poisson tail.

Player clean-sheet probability is derived from team clean-sheet probability and P60. Team features include venue, FDR, opponent attack, possession, and defensive xGA. DCP adds rolling tackles, interceptions, clearances, blocks, recoveries, and related features.

### Detailed logic

1. **Create one team-match row.** Player rows are merged with team form and collapsed to season/GW/team/venue/date. The clean-sheet label is (1[GA=0]); the goals-conceded regression target is team goals against.
2. **Build team features.** Venue, venue-specific FDR, opponent attack strength, team possession, and venue-specific defensive xGA (raw or standardized) form the common CS/GC feature vector.
3. **Fit team heads.** A LightGBM binary classifier estimates raw \(P(GA=0)\). Optional monotonic constraints can be applied, and an isotonic calibrator fitted on a chronological training tail transforms the raw CS probability. A separate LightGBM regressor estimates expected goals conceded and is clipped at zero.
4. **Join player exposure.** Evaluation players are joined to the minutes model. The preferred P60 column is a calibrated/raw 60-minute probability. If enabled and P60 is absent, the fallback is
   \[
   \tilde p_{60}=\operatorname{clip}\left(\frac{\hat M-30}{60},0,1\right).
   \]
   Player clean-sheet probability is then `p_teamCS × prob_played60_use`.
5. **Define defensive-contribution counts.** For defenders, the count is clearances + blocks + tackles + interceptions. Midfielders and forwards additionally include recoveries. These counts are divided by actual `minutes/90` to form training rates. Rows below `min_dcp_minutes` are excluded, and remaining rows receive exposure weights equal to minutes/90.
6. **Fit DCP rate models.** Separate LightGBM regressors for DEF, MID, and FWD predict non-negative contribution intensity \(\lambda_{90}\). No goalkeeper model is trained.
7. **Convert DCP rate to threshold probability.** Expected match intensity is \(\lambda=\lambda_{90}\hat M/90\). The threshold is also scaled by expected minutes and rounded upward:
   \[
   k=\left\lceil K_{90,pos}\frac{\hat M}{90}\right\rceil,
   \]
   with defaults of 10 for defenders and 12 for midfielders/forwards. The final DCP probability is the Poisson upper tail \(P(N\ge k\mid N\sim Poisson(\lambda))\); expected contributions equal \(\lambda\).
8. **Emit three contracts.** Output rows contain team CS probability, player eligibility-adjusted CS probability, expected GC, DCP probability, expected DCP count, minutes, and truth. CS, GC, and DCP metrics are written separately.

### Strengths

- Clean sheets are correctly modeled at team-match level rather than independently for every player.
- Player eligibility is separated through P60, avoiding full CS credit for likely substitutes.
- Team and player defensive tasks are separated, and DCP thresholds vary by position.
- Calibration artifacts and team/player metrics are produced.
- The goals-conceded head supports the nonlinear FPL concession penalty.

### Weaknesses

- Rolling team CS AUC of 0.532 is only marginally above random; historical player AUC of 0.704 did not carry forward.
- Rolling DCP AUC of 0.476 is worse than random ordering, and Brier 0.383 is poor. The persisted `metrics_dcp.json` is `{}`, so artifact completeness checks failed to catch an unevaluated submodel.
- The DCP implementation assumes Poisson counts despite overdispersion, player role dependence, and correlation among component actions.
- Goals conceded uses a generic regression objective and clipping rather than a count distribution; its saved MAE of 1.186 is large relative to typical team goals conceded.
- CS and GC are separately modeled and may be incoherent: in a single count model, `P(CS)` should equal `P(GC=0)`.
- The player-CS code may multiply a player-level `prob_cs` that already includes P60 by P60 again in some aggregation paths. The two schemas (`prob_cs` versus `p_teamCS`) make double exposure possible and require a contract test.
- Current rolling DCP validation depends on reconstructing the model's minutes-scaled threshold; official scoring semantics should be centralized and versioned.

### Improvements

1. Replace separate CS/GC heads with a calibrated Poisson, negative-binomial, or Dixon–Coles-style team score model; derive both expected GC and CS probability from the same distribution.
2. Benchmark against bookmaker clean-sheet odds and simple xGA/opponent-xG baselines.
3. Rebuild DCP as threshold classification or an overdispersed count/hurdle model, with position and role-specific calibration.
4. Fail training if any required metric file is empty, a class is absent, or rolling AUC/Brier is worse than the baseline.
5. Define one schema: `team_prob_cs` is team-level; `player_prob_cs_eligible` includes P60. Assert that aggregation applies exposure exactly once.
6. Add projected possession, opponent style, likely game state, aerial-duel volume, role, and starting lineup interactions.
7. Report team-level calibration by probability bin and DCP metrics by position and threshold prevalence.

## 4. Saves model

### Overview

[`saves_model_builder.py`](../src/fpl_assistant/models/saves_model_builder.py) trains goalkeeper saves-per-90 models using LightGBM and an optional Poisson/Tweedie head. Features cover venue, team defense, opponent attack, rolling saves, and shots on target faced. Per-match saves are scaled by predicted minutes.

### Detailed logic

1. **Restrict the universe.** Played-match player-form rows are normalized and reduced to goalkeepers with observed saves and positive minutes.
2. **Build the target.** The training response is
   \[
   y_{save90}=\frac{saves}{minutes/90}.
   \]
   Earlier seasons and pre-cutoff matches in the final season train the model; the final-season tail is held out.
3. **Build features.** The model uses venue, team defensive strength, opponent attacking strength, and sufficiently complete rolling home/away/all-venue goalkeeper saves-per-90 and shots-on-target-against-per-90 features.
4. **Fit the mean head.** A LightGBM L2 regressor is trained with the final 15% of chronological training rows as an early-stopping validation set, using L1 validation loss. Negative predictions are clipped to zero.
5. **Fit the optional count head.** A median imputer learned on training data feeds a Tweedie regressor with `power=1`, log link, and small L2 penalty. Its non-negative output is an alternative save rate.
6. **Scale by playing time.** The held-out goalkeeper row is joined to predicted minutes. For either head,
   \[
   \widehat{saves}=\widehat{saves}_{90}\frac{\hat M}{90}.
   \]
   Requiring predicted minutes turns any missing join into a failure; otherwise missing rows are audited.
7. **Evaluate exposure separately.** The builder reports per-90 MAE, match MAE using actual minutes, and match MAE using predicted minutes. This distinguishes rate-model error from minutes-model error.
8. **Persist the contract.** Both head outputs, true minutes/saves, feature schema hash, importances, metadata, model files, and missing-minutes audits are saved.

### Strengths

- Restricting the model to goalkeepers and separating per-90 rate from exposure is appropriate.
- Shots-on-target-against features are directly connected to the target.
- Both nonlinear and count-oriented heads are retained and evaluated.
- The saved evaluation explicitly tests actual-minutes and predicted-minutes scaling.
- Feature hashing, metadata, importances, and missing-minutes audits support reproducibility.

### Weaknesses

- Only 252 saved test rows and 181 rolling rows make conclusions unstable.
- The Poisson head is consistently slightly worse than the LightGBM mean head in the stored metrics.
- Rolling correlation of -0.016 indicates no ability to rank goalkeeper save totals, even though MAE appears superficially tolerable.
- Forecast GW5 is absent from the rolling saves sample, indicating orchestration/artifact coverage failure.
- The model uses predicted minutes but does not explicitly model starter identity; backup keepers create a structural zero-inflation problem.
- Saves are conditional on shots on target and goals; independently predicting saves can become inconsistent with the defense model.

### Improvements

1. Use a two-stage model: probability of starting, then shots on target faced and save probability conditional on starting.
2. Couple saves to the team match model: predicted opponent SOT minus expected goals provides a coherent structural estimate.
3. Compare with simple baselines (recent saves/90, opponent SOT average, bookmaker opponent goal total) and require positive rank skill.
4. Use negative-binomial or quasi-Poisson models if dispersion tests reject Poisson.
5. Add goalkeeper identity/team random effects and partial pooling to stabilize small samples.
6. Make missing forecast windows a hard pipeline failure.

## 5. Discipline model

### Overview

[`discipline_model_builder.py`](../src/fpl_assistant/models/discipline_model_builder.py) trains separate per-90 LightGBM regressors for yellow cards, red cards, own goals, and missed penalties, with optional Poisson/Tweedie heads. It scales event rates by expected minutes and can combine a defensive goals-conceded rate into negative FPL points.

### Detailed logic

1. **Resolve event columns.** The loader maps available aliases to yellow cards, red cards, own goals, and missed penalties. If a target column is absent, it is created as all zeros.
2. **Form per-90 responses.** For each event (e), the builder calculates \(y_{e,90}=count_e/(minutes/90)\); zero-minute and missing results are filled with zero. Only played rows enter model fitting.
3. **Build features.** Venue, FDR, lagged minutes, team defense, opponent attack, days since last match, active status, encoded position, and sufficiently complete discipline-related rolling features form (X). The position encoder is currently fitted on the full loaded frame.
4. **Choose training and target slices.** Evaluation mode holds out the last `test_last_n` gameweeks of `test_season`. Prediction mode keeps that training cutoff but selects any requested season/GWs as the inference target.
5. **Fit eight possible heads.** Four LightGBM regressors independently predict non-negative event rates. With `--poisson-heads`, four median-imputed Tweedie log-link regressors provide alternate rates.
6. **Resolve minutes robustly.** Target rows join on season/GW/date/player/team to the minutes prediction. Missing expected minutes are imputed hierarchically from training-only team-position-season, position-season, position, and global median minutes, multiplied by an analogous estimated probability of playing.
7. **Scale and score events.** Expected counts are \(\hat e=\hat e_{90}\hat M/90\). FPL deductions are
   \[
   -1\hat{YC}-3\hat{RC}-2\hat{OG}-2\widehat{MP}.
   \]
8. **Add goals-conceded deductions.** A team goals-against rate is read directly or inferred as \(-\log(P(CS))\). For GK/DEF exposure \(\lambda_{on}=\lambda_{GA}\min(\hat M/90,1)\), the expected number of complete conceded pairs is
   \[
   E[\lfloor N/2\rfloor]=\lambda_{on}/2-(1-e^{-2\lambda_{on}})/4,
   \]
   which is subtracted from expected points.
9. **Write outputs.** The artifact contains each per-90 rate, expected count, point deduction, total negative points, minutes, and playing probability. Evaluation MAEs are printed to logs but are not saved to a metrics file.

### Strengths

- It covers negative components often omitted from xPoints systems.
- Rate/exposure separation and the analytical expectation for each pair of goals conceded are conceptually sound.
- Training-only hierarchical imputation reduces direct leakage when minutes joins fail.
- The model provides join-audit output and flexible defense-schema ingestion.

### Weaknesses

- No metrics JSON is persisted; MAE is only logged. The existing artifacts cannot establish model quality.
- All three saved versions predict mean yellow cards and red cards of exactly zero. They also predict zero missed penalties. This is a degenerate failure for the most common discipline event.
- Rare event per-90 targets explode for low-minute appearances; training every played row without exposure weights makes these targets noisy.
- The feature artifacts in `v3` contain no historical card-rate features, only venue, FDR, previous minutes, team/opponent strength, availability, and encoded position.
- `OrdinalEncoder` is fitted before the split; although position is stable and low-cardinality, preprocessing should still be trained and saved on training data only.
- The active [`points_forecast.py`](../src/fpl_assistant/models/points_forecast.py) does not consume discipline predictions. It applies fixed per-position priors, leaving the trained model disconnected.
- The forecaster uses a different minutes schema (`pred_minutes`) from the discipline builder's expected `pred_exp_minutes`, increasing integration fragility.

### Improvements

1. Do not use the current trained outputs. Keep the transparent position priors until a discipline model beats them out of sample.
2. Persist count MAE, Poisson deviance, log loss/Brier for event occurrence, prevalence, calibration, and baseline comparisons for every head.
3. Model yellow cards first with a weighted binary or count model; retain priors for extremely rare RC/OG/missed-pen events unless there is enough data.
4. Weight rate training by minutes/exposure or model raw counts with a log-minutes offset.
5. Add referee, fouls, tackles, duel volume, opponent dribbles, position/role, derby intensity, and historical card-rate features.
6. Establish a single forecast contract and integrate `neg_points_total` into xPoints behind an evaluated feature flag.
7. Add a non-degeneracy gate: fail if a common-event head predicts an all-zero or near-zero distribution.

## 6. Expected-points model

### Overview

The current [`points_forecast.py`](../src/fpl_assistant/models/points_forecast.py) is a deterministic probabilistic scoring layer. It calculates appearance, goal/assist, clean-sheet, goals-conceded, saves, DCP, and discipline components, but its final total currently includes only the positive components. It uses exact expectations for grouped saves and goals-conceded penalties under Poisson assumptions. An older historical aggregator, [`expected_points_aggregator.py`](../src/fpl_assistant/models/expected_points_aggregator.py), uses similar logic but a different schema.

### Detailed logic

1. **Resolve and merge component windows.** Minutes is the base player-fixture universe. Goals/assists, defense, and saves are de-duplicated and merged on season, GW, player, and team, with nearest-date reconciliation when needed. Metadata such as fixture, opponent, venue, and FDR is coalesced from component files.
2. **Recover event intensities.** Goal and assist probabilities are converted to Poisson means with \(\lambda=-\log(1-p)\). If probabilities are missing, predicted count means are used. Position scoring is 6/6/5/4 points per goal for GK/DEF/MID/FWD and 3 points per assist.
3. **Calculate appearance points.** `p1` comes from `p_play`, falling back to (1[\hat M>0]). P60 falls back to `clip(pred_minutes/90,0,1)` and is forced not to exceed `p1`. Appearance xPoints is an upstream `exp_minutes_points` value when available, otherwise \(p1+p60\).
4. **Calculate clean-sheet points.** A selected defense probability is multiplied by P60 and the positional CS award: 4 for GK/DEF, 1 for MID, and 0 for FWD.
5. **Calculate concession deductions.** The code prefers a defense `lambda90`, then expected GC, then \(-\log(P(CS))\). It scales this by `pred_minutes/90`, evaluates the Poisson expectation of complete pairs conceded, and negates it for GK/DEF.
6. **Calculate save points.** A per-match saves mean is preferred; a per-90 value is exposure-scaled if necessary. For GK only, the code calculates the exact Poisson expectation \(E[\lfloor S/3\rfloor]\) and multiplies it by `p1`.
7. **Calculate discipline and DCP.** Discipline is currently a fixed per-90 position prior—GK -0.05, DEF -0.18, MID -0.14, FWD -0.10—scaled by expected minutes. DCP uses a supplied probability when present; otherwise it derives Poisson tail probabilities at 10 actions for DEF and 12 for MID/FWD from a contribution intensity. Expected DCP bonus is (2P(DCP)).
8. **Sum the current production total.** The actual code defines
   \[
   xPts=xP_{appearance}+xP_{goals}+xP_{assists}+xP_{CS}+xP_{saves}+xP_{DCP}.
   \]
   **Despite calculating and outputting them, `xp_concede_penalty` and `xp_discipline_prior` are omitted from this sum.** No bonus-points component is included. This is a correctness defect, not merely a modeling limitation.
9. **Publish rows and cumulative stack.** Component values and `xPts` are written to a GW-window CSV/Parquet and appended to the cumulative expected-points file. Missing defense or saves inputs default their contributions to zero after warning.

The older `expected_points_aggregator.py` differs materially: it includes concession penalties in its component sum, calculates DCP as `2 × prob_dcp`, and uses a five-column key including date. Its coexistence with the active forecaster explains some schema and result differences in historical artifacts.

### Strengths

- Component decomposition is transparent and auditable.
- Nonlinear scoring rules for saves and goals conceded are handled analytically rather than by crude division.
- Missing-input warnings, coverage reporting, forced keys, date reconciliation, and roster filtering are useful production guardrails.
- The output includes component xPoints, enabling attribution and debugging.
- Rolling total bias is modest (-0.30 points), and MAE of 1.95 is a reasonable starting point.

### Weaknesses

- Rolling correlation is only 0.214, which is inadequate for reliably ranking transfers, starters, and captains.
- MAE is dominated by the many low-scoring/DNP rows. There are no top-k, rank, regret, captain, or squad-level metrics in the artifact.
- No bonus-point model is present in the active forecaster, despite bonus being material and concentrated among premium picks.
- The trained discipline model is ignored.
- The active total omits the already-computed concession and discipline columns, systematically overstating relevant GK/DEF outcomes and making the published component breakdown inconsistent with `xPts`.
- Independent component expectations discard covariance among minutes, goals, assists, clean sheets, saves, bonus, and cards.
- The code supports many fallback column names and schema variants. This resilience can silently combine semantically different quantities—for example team versus player CS probability, and per-90 versus per-match save rates.
- The latest historical expected-points artifact (`v3`) has no actuals/metrics; `v2` has actuals but very low correlation (0.076). `v1` has a different universe and severe negative bias, so artifact lineage is not trustworthy.
- FDR conflict resolution by mode/tie-max is arbitrary and can hide upstream disagreement.

### Improvements

1. Define and validate a typed schema for every component, including unit (`per90`, `per_match`, probability), exposure status, model version, training cutoff, and calibration version.
2. Add a bonus model based on expected BPS or simulate bonus jointly within each fixture.
3. Produce player outcome distributions with Monte Carlo simulation, retaining component dependence and minutes uncertainty.
4. Evaluate Spearman rank correlation, top-10/25 precision, captain regret, XI regret, calibration by xPoints band, and realized squad points—not only row-level MAE.
5. Train a simple stacked/calibration layer on strictly out-of-fold component predictions to correct systematic biases while retaining interpretability.
6. Make missing DEF/SAV/discipline inputs configurable as hard failures in production rather than silently defaulting to zero.
7. Remove or formally deprecate the older aggregator once one contract becomes canonical.

## 7. Captain ranker

### Overview

[`captain_ranker.py`](../src/fpl_assistant/models/captain_ranker.py) uses LightGBM LambdaRank to rank players within each gameweek. Features are primarily downstream xPoints components, price, and position. It reports NDCG@5, precision@5, and whether the actual top scorer appears in its predicted top five.

### Detailed logic

1. **Build player-GW rows.** Expected-points fixture rows are summed to one player/gameweek row, so double gameweeks contribute combined component values. Actual FPL points are independently summed to the same season/GW/player key.
2. **Attach price and position.** The registry supplies the exact GW price where available; otherwise the last known prior price is used, with backward fill as a final fallback. Position is normalized and one-hot encoded.
3. **Construct features.** Available features are total xPoints, P60, playing probability, goal/assist/CS/save/concession components, price, and four position indicators. Player/team identifiers and raw actual points are excluded.
4. **Transform labels.** Missing actual points become zero and negative FPL scores are clipped to zero because the LambdaRank implementation requires non-negative relevance labels.
5. **Split by time.** Test rows are the requested contiguous gameweek window in the selected season. Training contains all earlier seasons and only gameweeks before that window in the test season.
6. **Group the ranking task.** Rows are sorted by season/GW/player and each season-gameweek becomes one LambdaRank query group. An 800-tree LightGBM model with the `lambdarank` objective learns a score whose ordering should maximize NDCG.
7. **Produce recommendations.** The ranker scores every test player and selects the five highest scores per gameweek. The score is ordinal and has no direct points interpretation.
8. **Measure ranking quality.** NDCG@5 is macro-averaged across gameweeks. Precision@5 is the overlap between predicted and realized top fives; hit@5 asks whether the single row selected as actual top scorer appears in the predicted five. A binary-classifier alternative labels the top five as positive and uses class-weighted LightGBM probabilities.

### Strengths

- A learning-to-rank objective matches the decision better than pointwise regression.
- Grouping by season/gameweek is correct for ranking.
- The current source uses a chronological test window and saves fold metadata.
- Ranking-specific metrics are more useful than MAE for captain selection.

### Weaknesses

- `v6` performance is weak: only 13.3% precision@5 and a 16.7% top-scorer hit rate across six GWs.
- Six gameweeks are far too few for stable captain evaluation, especially with tied actual points.
- `v5` fold metadata includes GWs 36–38 in training while testing GWs 30–35. Its better NDCG (0.460) is contaminated by future information. `v6` correctly lists only GW29 from the test season and performs worse (0.375).
- Negative actual scores are clipped to zero, changing within-GW relevance and erasing downside that matters for captaincy.
- `actual_top5` is selected by a simple sort; ties at the cutoff make precision/hit metrics arbitrary.
- The feature set mostly re-ranks xPoints and price. It does not add ceiling, penalty duty, captain ownership, opponent uncertainty, or outcome variance.
- The rank score is not an expected captain-points estimate and should not be fed into an optimizer as though it were cardinal.

### Improvements

1. Discard `v5` metrics and any version whose fold metadata includes dates after the test window.
2. Use season-long rolling-origin evaluation over multiple seasons, with bootstrap confidence intervals by GW.
3. Compare directly with ranking by xPoints, bookmaker anytime-scoring probability, and official FPL `xP`. Keep the ranker only if it adds incremental regret reduction.
4. Use tie-aware relevance and report captain regret: actual points of hindsight-best captain minus actual points of selected captain.
5. Add predicted ceiling probabilities (for example, P(points ≥ 8/10/15)), penalty share, start probability, fixture/team goal distribution, and uncertainty.
6. Consider optimizing expected doubled points directly from simulated distributions instead of a separate ranker.

## 8. Squad and transfer optimizers

### Overview

The repository contains several optimization generations:

- [`squad_optimizer.py`](../src/fpl_assistant/models/squad_optimizer.py): single-GW initial squad;
- [`three_gw_optimizer.py`](../src/fpl_assistant/models/three_gw_optimizer.py): transfer-aware multi-GW squad planning;
- [`single_gw.py`](../src/fpl_assistant/optimizers/single_gw.py) and [`multi_gw.py`](../src/fpl_assistant/optimizers/multi_gw.py): newer MILP decision engines;
- [`multi_gw_hold.py`](../src/fpl_assistant/optimizers/multi_gw_hold.py), [`simulator.py`](../src/fpl_assistant/optimizers/simulator.py), and [`strategy.py`](../src/fpl_assistant/optimizers/strategy.py): holding/chip, simulation, and risk-aware strategy layers.

They use binary/integer programming to maximize expected points subject to budget, club, lineup, formation, transfer, captain, bench, and—in newer paths—chip constraints. Risk penalties and Monte Carlo strategy metrics are available in parts of the newer stack.

### Detailed logic

There are two materially different generations, so their logic should not be conflated.

**Legacy single-GW squad builder (`models/squad_optimizer.py`)**

1. Filter expected-points rows to one season/GW and normalize price heuristically: median price ≥25 means tenths and is divided by ten; otherwise values are treated as millions.
2. Create binary `pick_i` and `start_i` variables. Maximize
   \[
   \sum_i xP_i start_i+w_bxP_i(pick_i-start_i).
   \]
3. Require 15 picks, 11 starters, `start_i ≤ pick_i`, two total goalkeepers, one starting goalkeeper, minimum XI counts of 3 DEF/2 MID/1 FWD, budget, club cap, and a minimum expected-minutes eligibility threshold for starters.
4. It does **not** require exact full-squad DEF/MID/FWD quotas, which explains the invalid saved squads.

**Legacy three-GW planner (`models/three_gw_optimizer.py`)**

1. Aggregate double-gameweek xPoints per player/GW and join a price/availability registry for every horizon week.
2. Create per-GW squad, XI, captain, buy, and sell variables. Enforce exact squad quotas, legal formations, club cap, budget, availability, and at most the configured transfers between adjacent weeks.
3. Maximize horizon XI xPoints plus discounted bench xPoints and captain uplift, with optional minimum-spend and minimum-number-of-high-xPoints-starters guardrails.

**Current single/multi-GW optimizers (`optimizers/`)**

1. Normalize a current team state, owned purchase/sale values, bank, free transfers, availability, player EV, captain uplift, and optional variance estimates.
2. Use binary variables for squad, XI, ordered bench, captain, vice-captain, buys/sells, and chip state; multi-GW models add bank flow, free-transfer rollover, hit counts, and squad transition equations.
3. Enforce exact 2/5/5/3 squad composition, legal XI formations, one captain and vice in the XI, club cap, affordability using sale proceeds, and chip-specific rules. Bench Boost activates bench EV; Triple Captain adds an extra captain multiple; Wildcard suppresses hit costs. Some variants also model Free Hit with a temporary squad.
4. A representative multi-GW objective is
   \[
   \sum_g\left(EV_{XI,g}+EV_{BB,g}+EV_{captain,g}+EV_{TCextra,g}
   -\lambda_{risk}Var_{XI,g}-4\,hits_g\right).
   \]
   This is a linear proxy: individual variance terms are summed without covariance.
5. After solving with CBC, the system extracts squads, formations, bench order, captains, chips, transfers, bank, free transfers, hit costs, and per-GW objective components.

**Simulation and strategy selection**

1. The simulator derives or respects an XI and bench order, then samples player/team outcomes from the component forecasts, including minutes and autosub behavior.
2. Strategy evaluation tests captain candidates with common random numbers, sums simulated team points over the horizon, subtracts transfer hits, and reports EV, standard deviation, 10% VaR, 10% CVaR, and probability of beating score thresholds.
3. Candidate strategies are normally ranked by EV with lower standard deviation as a tie-break; an alternate path can rank by downside-tail performance.

### Strengths

- MILP is the right tool for exact FPL roster and transfer constraints.
- The newer optimizer code models purchase/sale prices, transfer hits, captain/vice, bench order, chips, and multi-week state.
- Risk-aware objectives and scenario simulation are present, a strong foundation for decisions under uncertainty.
- Outputs include diagnostics and universe information rather than only a player list.

### Weaknesses

- Saved `squad_optimizer` outputs are invalid under standard FPL squad rules: positional counts include six or seven midfielders and only two forwards. The old optimizer constrains two total GKs and XI formation, but not exact 15-man squad quotas of 2 GK, 5 DEF, 5 MID, and 3 FWD.
- The latest saved summary records `budget_m: 1000.0` and a total price of £81.2m. A budget intended in tenths was apparently passed to an interface documented in millions, making the budget constraint ineffective.
- The objective values (for example 115.44 in GW34) are in-sample expected values, not realized backtest performance.
- Linear variance penalties sum individual variances and omit player correlations, team/fixture covariance, and captain covariance.
- Fixed bench weights do not reflect substitution probability, bench order, formation legality after no-shows, or correlated rotation.
- The older and newer optimizer implementations coexist without a clear canonical path, raising the risk that the app invokes obsolete logic.
- No saved transfer/chip backtest, regret analysis, or solver optimality history was found.

### Improvements

1. Retire or block `squad_optimizer.py`; use one canonical optimizer with exact squad quotas and explicit price units.
2. Add mandatory post-solve invariants for squad composition, XI formation, team cap, budget, captain/vice membership, distinct bench slots, transfers, bank evolution, and chip rules. Run these in pytest and in production before publishing a plan.
3. Replace heuristic price auto-detection with a typed input contract (`price_tenths` or `price_millions`) and reject implausible budgets.
4. Backtest complete weekly decisions against baselines: no-transfer, greedy xPoints, template team, and hindsight-optimal. Report realized points, hits, regret, rank, and turnover.
5. Feed joint Monte Carlo scenarios into strategy evaluation so correlations and autosub rules are represented.
6. Model transfer value and uncertainty over longer horizons; penalize fragile low-minutes squads and excessive churn.
7. Version the optimizer together with the exact prediction snapshot and team state used for each recommendation.

## Architecture and governance findings

### What is working

- Clear component separation and rich intermediate outputs make the system explainable.
- Most builders use chronological holdouts, deterministic seeds, version directories, and metadata.
- Audit files for missing joins/features are unusually good for a research project.
- The repository already contains the skeleton of rolling backtests, Monte Carlo simulation, and optimizer invariants.

### Systemic weaknesses

- **Fragmented code paths:** near-identical model modules exist under both `scripts/models` and `src/fpl_assistant/models`, while old and new aggregators/optimizers coexist.
- **Weak registry semantics:** latest pointers can reference incomplete or inferior versions, naming conventions vary (`latest_version.txt` versus `LATEST_VERSION.txt`), and defense has a different directory layout.
- **Evaluation inconsistency:** models use different test windows and schemas; several artifacts omit metrics, baselines, uncertainty, or provenance.
- **Silent fallbacks:** missing model inputs often become zeros or priors. This preserves pipeline completion at the cost of potentially invalid recommendations.
- **No end-to-end promotion gate:** component improvements are not required to improve player ranking, captain regret, or squad outcomes.
- **Point estimates dominate:** downstream decision layers need distributions and correlations, especially for captaincy, benching, and chips.

## Prioritized improvement plan

### P0 — correctness and safety

1. Fix the active `xPts` sum so it includes the computed concession and discipline deductions, then regression-test every FPL scoring component against hand-calculated fixtures.
2. Enforce exact optimizer invariants and typed price units; invalidate all non-compliant saved squad outputs.
3. Build an atomic model registry/manifest. Promote `latest` only after artifact completeness, schema, cutoff, smoke, and metric gates pass.
4. Fix the incomplete goals/assists `v5` latest pointer and choose an evaluated expected-points champion.
5. Make empty metrics, missing forecast GWs, degenerate predictions, and semantic unit mismatches hard failures.
6. Declare one canonical package path and deprecate duplicate/legacy model and optimizer entry points.

### P1 — evaluation

1. Run the existing backtest harness weekly across at least two complete seasons using deadline-stamped prediction snapshots.
2. Add simple baselines for every component and report skill relative to baseline with confidence intervals.
3. Add end-to-end ranking and decision metrics: Spearman correlation, top-k recall, captain regret, XI regret, transfer regret, and realized squad points.
4. Store per-fold predictions and manifests so every metric can be reproduced.

### P2 — modeling

1. Repair minutes gating and add lineup/news/congestion features.
2. Unify clean-sheet and goals-conceded modeling; rebuild DCP and saves with coherent count/threshold models.
3. Add an expected bonus/BPS component.
4. Either rebuild and integrate discipline or intentionally retain documented priors.
5. Generate joint player-point distributions and use them in the risk-aware optimizer.

## Recommended promotion criteria

A candidate model should not become latest unless it satisfies all of the following:

- identical rolling folds and row universe versus the current champion;
- no feature timestamp later than the prediction deadline;
- complete model, preprocessing, feature-schema, metadata, prediction, and metrics artifacts;
- improvement over a simple baseline and no material regression in protected slices;
- calibrated probabilities within agreed thresholds;
- end-to-end xPoints/ranking improvement where applicable;
- optimizer plans pass every invariant;
- uncertainty intervals show the apparent improvement is credible.

Suggested initial gates are illustrative and should be calibrated from historical folds: minutes MAE below 22 with P60 Brier below 0.16; positive saves and DCP rank correlation/AUC materially above 0.5; captain regret better than raw xPoints; and zero invalid optimizer plans.

## Final assessment

The architecture is directionally strong and the attacking model is a credible foundation. However, current saved artifacts show that model governance, minutes estimation, defense/saves generalization, and optimizer correctness are not yet strong enough for high-confidence autonomous FPL decisions. Fixing correctness and evaluation infrastructure will produce more value than adding model complexity. Once those foundations are in place, the project is well positioned to benefit from joint probabilistic simulation and risk-aware multi-gameweek optimization.
