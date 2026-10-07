# FPL Player Archetype Scheme — Implementation Specification (v1.0)

This document contains **46 complete specifications**: 18 main specifications covering 22 named archetypes/states, plus 28 position-specific Production Style composites. This represents **50 named user-facing labels/states** because the five mutually exclusive Usage states share one specification.

**How to read this document:** Owner choices are selected and the catalogue is the V1 implementation specification. Values marked as initial or search-grid parameters must be selected through chronological walk-forward validation and then recorded in the released model configuration. Deferred goalkeeper components remain outside V1.

## Selected owner choices

### CHOICE DATA-01 — Data source and field definitions

- **Selected — Existing project provider mapped to a canonical dictionary:** Keep the current providers [FPL, WhoScored, Understat], document every field name, unit, event definition, and known gap in one data dictionary. Disable a metric when the provider cannot supply it reliably.


Official FPL: FPL points, minutes, and official availability flags.
Understat: xG, npxG and xA.
WhoScored: shots, chances, defensive event statistics and detailed injury/availability information when present.
Internal registry: canonical player, club and match IDs plus listed position, price snapshots and mapped source identifiers and immutable source snapshots; 
Source priority: use the named canonical source for each metric rather than a universal provider priority.
Missing-season and missing-player behavior: return `no_score`, record the missing source/field reason and suppress any label requiring that core field.


### CHOICE FIT-01 — How component weights are finalized

- **Selected — Transparent fixed weights, then validate:** Start with the documented weights in this catalogue and keep them unless testing shows a clear problem. Easiest to explain and audit.


### CHOICE TIME-01 — How old and recent evidence is blended

- **Selected — Fixed family weights:** Use 25% previous-season evidence, 30% earlier-current-season evidence, and 45% from the latest 10 eligible appearances for every non-usage archetype.

The temporally combined score is:

$$
S_{\text{temporal}}
=
\sum_{i\in A}w'_iS_i
$$

where:

$$
w'_i=\frac{w_i}{\sum_{j\in A}w_j}
$$

and \(A\) is the set of available evidence windows.

If a window is unavailable, remove it and proportionally redistribute its weight across the available windows. Do not treat this redistribution as small-sample shrinkage; shrinkage is applied separately after temporal combination.

A window is available when it contains at least one eligible appearance and all core fields required by the archetype. A window with limited evidence remains available, but its uncertainty is handled by small-sample shrinkage and confidence rather than by changing its temporal weight.


### CHOICE VALID-01 — Minimum validation bar for release

- **Selected — Predictive improvement:** Require at least 2% improvement over the strongest simple baseline on the stated target, improvement in at least 70% of walk-forward folds, and no material calibration deterioration.


### CHOICE GK-01 — Goalkeeper production-style scope

- **Selected — Shot Stopper only in V1:** Implement Saves Machine/Shot Stopper now; defer Sweeper and Distributor components and every composite requiring W or P until their base specifications exist.


### CHOICE VALUE-01 — Main value view

- **Selected — Two separate views:** Show delivered historical value and expected next-five-gameweek value separately; never blend actual and forecast points into one number.


## Field glossary

Plain-language meaning of every field, for reference while reading the entries below.

- **ID:** A short, permanent code name used by the software that should not change even if the display name changes.
- **Display name:** The user-facing name shown in the app.
- **Family:** The broader category the archetype belongs to (Production Style, Usage, Fixture Behaviour, Venue Behaviour, Return Shape, or Value).
- **Definition in one sentence:** A clear description of what the label means.
- **V1 status:** Whether the archetype ships in the first version: KEEP V1, CONSIDER V1, DEFER, or REMOVE.
- **Eligible FPL positions:** Which positions can receive the label (GK, DEF, MID, FWD).
- **Eligible player population:** Which players are allowed into the calculation at all (e.g. active PL players with a minutes floor).
- **Required raw fields and provider definitions:** The original data needed to calculate the score and exactly what each field means per its data provider.
- **Metric directions (higher/lower is better):** Whether a higher or lower raw value represents more of the archetype.
- **Rate denominators (per 90/per appearance/per start/etc.):** How totals are made comparable across players with different minutes (per 90, per appearance, per start, per team match).
- **Context adjustments:** Factors controlled for so comparisons are fair (opponent strength, venue, red cards, game state, team strength, minutes played).
- **Metric preprocessing/caps:** How raw data is cleaned before scoring — missing values, outlier caps, skew transforms.
- **Component standardization:** How different statistics are put on a comparable scale before combining (e.g. position-relative z-scores or percentiles).
- **Component weights or fitted method:** How much each metric contributes — fixed weights or a fitted/learned method.
- **Raw score equation:** The exact formula combining the inputs before converting to the user-facing score.
- **Final score scale:** The range shown or stored after calculation.
- **Reference population:** The players someone is compared against (e.g. a midfielder vs. eligible midfielders, not all players).
- **Historical window:** Older evidence used to establish baseline, normally the previous season.
- **Earlier-current-season window:** Current-season appearances older than the recent window; useful but weighted less than recent form.
- **Recent window:** The newest evidence, weighted most heavily.
- **Temporal combination formula:** The exact method for blending historical, earlier-current, and recent evidence.
- **Minimum total minutes:** The smallest combined minutes required before the archetype is calculated/displayed reliably.
- **Minimum eligible appearances:** The minimum number of appearances meeting the appearance rule (e.g. 10 appearances of ≥30 minutes).
- **Minimum starts (if relevant):** The minimum starts required when starts matter (e.g. Undroppable, Regular Starter).
- **Minimum context samples (if relevant):** Evidence required within a specific situation (e.g. ≥8 home and ≥8 away appearances for a venue label).
- **Small-sample shrinkage:** A method pulling uncertain scores toward a sensible average so a hot 120-minute sample doesn’t outrank an established 2,500-minute player.
- **Label qualification threshold:** The score a player must reach before receiving the label.
- **Exit threshold/hysteresis:** A slightly lower threshold for removing a held label, so labels don’t flicker on/off around one cutoff.
- **Confidence formula and display bands:** How evidence strength is calculated and shown as Low/Medium/High Confidence.
- **Suppression rule at low confidence:** What happens when evidence is insufficient (e.g. show the score but hide the label until confidence clears a bar).
- **Trend formula and thresholds:** How recent behaviour is compared to the older baseline to call a trend.
- **Allowed statuses:** The status words that may accompany the archetype (Emerging, Established, Stable, Declining, Provisional, Insufficient Evidence).
- **Within-family exclusivity/precedence:** Whether a player can hold multiple labels from the same family, and which wins when rules overlap.
- **Transfer/new-club handling:** What happens when a player changes clubs — individual ability carries over, club-dependent evidence is discounted and confidence lowered.
- **Position-change handling:** What happens when FPL reclassifies a player — raw performances are retained but re-benchmarked against the new position.
- **Injury/absence handling:** How injuries, suspensions, omissions, and long gaps affect score and confidence, without assuming an absence means declining ability.
- **Missing-data fallback:** What happens when a required statistic is unavailable (reduced formula, prior, retain previous score, or no score) — never silent.
- **Output fields:** The information the calculation returns (player ID, archetype ID, score, active label, confidence, trend, status, evidence minutes, model version).
- **Unit tests:** Automated checks proving the calculation behaves correctly (e.g. a below-minutes player can’t get a full-confidence label).
- **Backtest target and acceptance criterion:** How the archetype is validated on unseen historical data and what counts as success.
- **Owner/version/effective date:** Who approved the definition, its version, and when it became active — the audit trail for formula changes.

## Resolved catalogue discrepancies

- **All-Rounder / Jack of All Trades:** Do not calculate it as a separate V1 model. Treat it as a derived display label when all relevant production components qualify; standalone modelling is deferred.
- **Anywhere Threat:** Keep in V1 as the venue-equivalence label. Home and away adjusted performance must be within ±0.20 standard deviations.
- **Bench Auto-fill:** Keep in V1 to complete the value/playing-time matrix; it still requires the minimum evidence floor.
- **Points Hazard:** The final display name is **Points Hazard**; **Discipline Risk** remains an alias.
- **Forward combinations:** **Creative Forward** is C only; **Complete Forward** is G+C. The duplicate G+C identifier is corrected below.
- **Complete Midfielder:** The component code is G+C+D.
- **Return shape overlap:** Explosive Returner and Steady Returner are not mutually exclusive. A player may be both frequently reliable and capable of large hauls.


### Production-style composite display rule

- Calculate and retain every eligible base production-component score and active/inactive state internally.
- Show **exactly one** production-style composite label to the user for the player's current FPL position.
- Choose the most specific active combination in this order: three-component label, then two-component label, then single-component label.
- A more specific label replaces its subset labels in the user-facing display. For example, a FWD with active G and C components displays **Complete Forward**, not Finisher + Creative Forward + Complete Forward.
- Apply each base component's entry/exit hysteresis before selecting the composite. When a component exits, immediately remap the player to the next most specific valid combination.
- Composite score and confidence remain the minimum score and minimum confidence among the required active components. Individual component scores, confidence and trends remain available in the output for explanation and modelling.
- Ignore deferred or unavailable components when selecting a V1 composite. Therefore, goalkeeper W/P combinations cannot display while those components remain deferred.
- This display exclusivity applies only to Production Style composites; it does not suppress labels from Usage, Fixture, Venue, Return Shape, Value or Risk families.
- **Goalkeeper combinations:** W and P do not have V1 base specifications. Every composite requiring W or P is `DEFER V1` and must not be calculated or displayed.


## Confidence bands display defaults

Bands:

- Insufficient <0.35; 
- Low 0.35–0.54; 
- Medium 0.55–0.74; 
- High ≥0.75.
- minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
- appearance_adequacy = min(eligible_appearances / required_appearances, 1)
- context_coverage = minimum proportion of required context samples met
- precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
- data_quality = available planned component weight / total planned weight

Precision defaults:

- Position-percentile scores: maximum_useful_width = 40 score points.
- Probabilities: maximum_useful_width = 0.40.
- Standardized fixture/venue effects: maximum_useful_width = 1.00 SD.
- Start probability: maximum_useful_width = 0.40


### Attack/defence update method

Use a sequential weighted Bayesian/MAP update after each completed match. Store the pre-match ratings first; the current match may only affect the next snapshot.

For home team $i$ against away team $j$: use shared attack/defence ratings with separate league baselines for xG and goals:

$$
\lambda^{xG}_{i,j,t}=\exp\left(\mu_{xG}+h_t+A_{i,t}-D_{j,t}\right)
$$

$$
\lambda^{G}_{i,j,t}=\exp\left(\mu_G+h_t+A_{i,t}-D_{j,t}\right)
$$

Use $h_t = \text{team\_model\_home\_log\_effect}$ for the home team and $h_t=0$ for the away team. The initial home log effect is `0.15`; select the released value from `0.05, 0.10, 0.15, 0.20, 0.25` through walk-forward validation. This parameter is separate from Elo home-advantage points.

Define the raw per-match losses as:

$$
\mathcal L^{raw}_{xG}
=
\frac{1}{2}\sum_{k\in\{home,away\}}
\left[\log(1+xG_k)-\log(1+\lambda^{xG}_k)\right]^2
$$

$$
\mathcal L^{raw}_{G}
=
\frac{1}{2}\sum_{k\in\{home,away\}}
\left[\lambda^G_k-G_k\log(\lambda^G_k)\right]
$$

The omitted $\log(G_k!)$ term is constant with respect to the ratings. Normalize the two losses using the corresponding league-average-model loss calculated on the training fold:

$$
\mathcal L_{xG}=\frac{\mathcal L^{raw}_{xG}}{\max(B_{xG},10^{-8})}
\qquad
\mathcal L_G=\frac{\mathcal L^{raw}_G}{\max(B_G,10^{-8})}
$$

where $B_{xG}$ and $B_G$ are frozen before scoring the validation fold. The match loss is:

$$
\mathcal L_{match}=0.70\mathcal L_{xG}+0.30\mathcal L_G
$$

Regularize new ratings toward the previous pre-match ratings using:

$$
\mathcal P_t
=
\frac{m_{prior}}{2}
(\theta_t-\theta_{t-1})^\top
\bar{I}_{t-1}
(\theta_t-\theta_{t-1})
$$

where $\theta$ contains all attack and defence ratings, $\bar I_{t-1}$ is the diagonal average per-match observed-information matrix from the training fold with every diagonal element floored at $10^{-6}$, and `m_prior` is the prior-equivalent match count. Use `m_prior = 8` initially and select from `4, 8, 12, 16` through walk-forward validation.

The sequential update is:

$$
\theta_t
=
\arg\min_{\theta}
\left[
\mathcal L_{match}(\theta)+\mathcal P_t(\theta,\theta_{t-1})
\right]
$$

Enforce identifiability after every update:

$$
\sum_i A_{i,t}=0
\qquad
\sum_i D_{i,t}=0
$$

Do not use $0.70xG+0.30G$ as a direct Poisson response. Record the selected loss baselines, prior strength and home coefficient in the model configuration.

## Team attack/defence strength V1

The intended flow is:

$$
\text{Player archetype} + \text{role/minutes} + \text{team attack} + \text{opponent defence}
\rightarrow \text{fixture-specific FPL expectation}
$$

For defensive players and goalkeepers:

$$
\text{Player archetype} + \text{role/minutes} + \text{team defence} + \text{opponent attack}
\rightarrow \text{clean-sheet and defensive-return expectation}
$$

The framework keeps three related but distinct concepts:

1. **Overall Elo**: a common baseline for general team quality.
2. **Attacking strength**: how well a team creates and converts attacking opportunities.
3. **Defensive strength**: how well a team prevents high-quality opposition opportunities.

These dimensions should be connected, but not treated as interchangeable.

---

### 1. Overall Elo as the Common Baseline

Overall Elo gives every team a shared rating on the same scale. It provides a stable starting point because it uses match results to summarize broad team quality.

Let team $i$'s overall rating before match $t$ be $E_{i,t}$. Against team $(j)$, its expected result can be written as:

$$
p_{i,t}=\frac{1}{1+10^{-\left(E_{i,t}-E_{j,t}+H\right)/400}}
$$

where $H$ is home advantage on the Elo scale. If the actual match score $s_{i,t}$ is $1$ for a win, $0.5$ for a draw, and $0$ for a loss, the update is:

$$
E_{i,t+1}=E_{i,t}+K_E\left(s_{i,t}-p_{i,t}\right)
$$

- `elo_home_advantage_points = 36` initially; apply `0` for the away team and select the released value from `36, 40, 60, 80` through walk-forward validation.
- `elo_update_k = 20` initially; select the released value from `10, 20, 30, 40` through walk-forward validation. This is an Elo rating-step parameter, not an optimizer learning rate.
- `team_model_home_log_effect = 0.15` initially; select the released value from `0.05, 0.10, 0.15, 0.20, 0.25` through walk-forward validation.

Overall Elo is useful, but insufficient for FPL modelling. Two teams can have similar overall ratings while reaching that level differently: one through an elite attack and average defence, another through an average attack and elite defence. They should not create identical expectations for attackers, defenders, or goalkeepers.

Overall Elo therefore acts as the **common prior and general-quality anchor**, while attacking and defensive ratings provide specialist dimensions.

#### Avoiding double-counting

Do not mechanically add overall Elo to attacking and defensive strength when all three already contain the same information. Use one controlled design:

1. **Hierarchical prior**: overall Elo initializes or shrinks attack and defence ratings, while predictions use the specialist ratings.
2. **Residual model**: attack and defence ratings measure deviations from what overall Elo predicts.
3. **Estimated blend**: all ratings enter the prediction, but their coefficients are learned and tested out of sample.

The hierarchical prior is the clearest default. Elo supplies stability, especially early in a season, while specialist evidence separates teams with different underlying processes.

---

### 2. Separate Attacking and Defensive Dimensions

For each team \(i\), maintain:

$$
A_{i,t}=\text{latent attacking rating}
$$

$$
D_{i,t}=\text{latent defensive rating}
$$

Use a consistent sign convention:

- higher \(A\) means a stronger attack;
- higher \(D\) means a stronger defence.

For display, transform the latent ratings into indices centred at league average:

$$
AS_{i,t}=\exp\left(\frac{A_{i,t}}{c_A}\right)
$$

$$
DS_{i,t}=\exp\left(\frac{D_{i,t}}{c_D}\right)
$$

Here, \(AS=1.00\) and \(DS=1.00\) are league average; values above one are stronger. Calling \(DS\) a **defensive resistance index** can reduce sign confusion.

- ${c_A} = {c_D} = 1$


#### Fixture interaction

The attacking expectation for team \(i\) against opponent \(j\) should depend on both dimensions:

$$
\eta_{i,j,t}=\mu+h_{i,t}+A_{i,t}-D_{j,t}
$$

where \(\mu\) is the league baseline and \(h_{i,t}\) is home advantage. With a log link for goals or xG:

$$
\lambda_{i,j,t}=\exp\left(\eta_{i,j,t}\right)
$$

### Promoted-team initialization
Use the following selected rule for a newly promoted club's first Premier League ratings.

- **Premier League average:** Start promoted teams at league average with very low confidence, then update rapidly. Neutral, but usually too optimistic.

### Season-start regression
Use the following selected season-start regression.

- **Retain 70% and regress 30% to league average:** Preserves genuine strength while allowing for transfers, managers and tactical changes.
$$
A_{\text{new}}=0.70A_{\text{old}}+0.30A_{\text{league}}
$$
$$
D_{\text{new}}=0.70D_{\text{old}}+0.30D_{\text{league}}
$$

### Attack/defence update strength
Use the following selected update strength.

- **Prior equivalent to eight matches:** Begin with the existing rating carrying the weight of eight matches. Every new match gradually reduces that prior’s influence.

### Match-result and xG weighting

Apply the global **Attack/defence update method**: normalized match loss is 70% xG loss and 30% goal loss, with the prior equivalent to eight matches initially. Never combine xG and goals into a direct Poisson response. Match results update overall Elo only and must not be added again to the specialist attack/defence objective.

### Pre-match rating timestamp
Use the following selected snapshot rule.

- **Immediately before the match, before its data is processed:** Includes every earlier match but none of the current match.


### Parameter search ranges
Use the following selected tuning method.

- **Small controlled walk-forward grid:** Test a limited set of sensible values on chronological validation seasons.

| Parameter | Values to test |
|---|---|
| Season retention | `0.60, 0.70, 0.80` |
| Prior-equivalent matches | `4, 8, 12, 16` |
| xG weight | `0.50, 0.70, 0.85, 1.00` |
| Goal weight | `1 - xG weight` |
| `elo_update_k` | `10, 20, 30, 40` |
| `elo_home_advantage_points` | `36, 40, 60, 80` Elo points |
| `team_model_home_log_effect` | `0.05, 0.10, 0.15, 0.20, 0.25` |
| MAP `m_prior` | `4, 8, 12, 16` equivalent matches |


---

## Part 1 — Main archetype catalogue (18 specifications; 22 labels/states)

### GOAL_THREAT — Goal Threat

- **ID:** GOAL_THREAT
- **Display name:** Goal Threat
- **Family:** Production style
- **Definition in one sentence:** Scores goals at a rate that exceeds position-relative expectation, combining underlying process (shots/xG) with actual outcomes.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF (position-relative; not restricted to MID/FWD)
- **Eligible player population:** Outfield players with sufficient minutes; excludes GKP
- **Required raw fields and provider definitions:** minutes; non-penalty expected goals (npxG); shots on target; shots in the box; non-penalty goals. Penalties are excluded. Provider-specific definitions follow CHOICE DATA-01.
- **Metric directions (higher/lower is better):** Higher
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate minutes and event totals; winsorize per-90 components at the 1st/99th position-season percentiles; use explicit missing flags and never replace missing events with zero unless the provider defines zero.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** V1 fixed weights: 45% npxG/90, 25% shots-in-box/90, 20% shots-on-target/90, 10% non-penalty-goals/90. Use these fixed weights under CHOICE FIT-01; change them only in a later version after walk-forward validation.
- **Raw score equation:** `0.45*z(npxG90) + 0.25*z(shots_in_box90) + 0.20*z(SOT90) + 0.10*z(npG90)`, with each z-score calculated within FPL position.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF (position-relative; not restricted to MID/FWD)) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Enter at the 80th percentile within FPL position unless an archetype-specific practical-effect rule is stricter.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. 
    Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight

- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Not mutually exclusive within Production style family; combines with Creator/Defensive Engine per position (see Production style naming system)
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict next-10-appearance npxG/90 and goal-return probability better than goals/90 alone across at least three walk-forward seasons; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### CREATOR — Creator

- **ID:** CREATOR
- **Display name:** Creator
- **Family:** Production style
- **Definition in one sentence:** Creates high-quality scoring chances for teammates at a rate exceeding position-relative expectation, based on xA/chance creation rather than raw assists.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF (position-relative)
- **Eligible player population:** Outfield players with sufficient chance-creation involvement; excludes GKP
- **Required raw fields and provider definitions:** minutes; expected assists (xA); key passes/chances created; big chances created; shot-creating actions (SCA); assists for display only. Exact provider definitions follow CHOICE DATA-01.
- **Metric directions (higher/lower is better):** higher is better
- **Rate denominators (per 90/per appearance/per start/etc.):** per 90 of appearances
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** V1 fixed weights: 50% xA/90, 25% key passes/90, 15% big chances created/90, 10% SCA/90. Assists are displayed but excluded from the core score to reduce teammate-finishing luck. Use these fixed weights under CHOICE FIT-01; change them only in a later version after walk-forward validation.
- **Raw score equation:** `0.50*z(xA90) + 0.25*z(key_passes90) + 0.15*z(big_chances_created90) + 0.10*z(SCA90)` within FPL position.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF (position-relative)) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Use fixed 25/30/45 weights when all three windows contain evidence. If a window is unavailable, redistribute its weight proportionally across the available windows. Small-sample shrinkage is applied after temporal combination and is separate from temporal weighting.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Enter at the 80th percentile within FPL position unless an archetype-specific practical-effect rule is stricter.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`.
    Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Not mutually exclusive within Production style family; combines with Goal Threat/Defensive Engine per position
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict next-10-appearance xA/90 and assist-return probability better than assists/90 or key-passes/90 alone; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### DEFENSIVE_ENGINE — Defensive Engine

- **ID:** DEFENSIVE_ENGINE
- **Display name:** Defensive Engine
- **Family:** Production style
- **Definition in one sentence:** Contributes high-value defcon/ground-covering actions at a rate exceeding position-relative expectation.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** MID, DEF, FWD
- **Eligible player population:** Outfield players with sufficient minutes; excludes GKP
- **Required raw fields and provider definitions:** minutes; tackles won; interceptions; clearances; blocks; recoveries for MID/FWD; and DefCon threshold hits (10 qualifying actions for DEF, 12 for MID/FWD, subject to current FPL rules). Definitions follow CHOICE DATA-01.
- **Metric directions (higher/lower is better):** higher is better
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** DEF: 18.75% each tackles won, interceptions, clearances and blocks, plus 25% DefCon hit rate. MID/FWD: 15% each tackles won, interceptions, clearances, blocks and recoveries, plus 25% DefCon hit rate. Use these fixed weights under CHOICE FIT-01; change them only in a later version after walk-forward validation.
- **Raw score equation:** Weighted sum of position-standardized action rates and DefCon hit rate using the listed position-specific weights.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Eligible players in the same FPL position: DEF, MID or FWD. FWD eligibility is retained because forwards can qualify for the FPL defensive-contribution threshold.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Enter at the 80th percentile within FPL position unless an archetype-specific practical-effect rule is stricter.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Not mutually exclusive within Production style family; combines with Goal Threat/Creator per position
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict next-10-appearance defensive-contribution points/returns better than total defensive actions/90 alone; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### CLEAN_SHEET_SPECIALIST — Clean-Sheet Specialist

- **ID:** CLEAN_SHEET_SPECIALIST
- **Display name:** Clean-Sheet Specialist
- **Family:** Production
- **Definition in one sentence:** Likelihood to get clean sheets, distinct from the team’s overall defensive forecast.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** DEF, GKP
- **Eligible player population:** Defenders and goalkeepers with sufficient appearances
- **Required raw fields and provider definitions:** minutes; starts; clean sheets while on pitch; goals conceded while on pitch; team expected goals against (xGA); opponent attacking strength; team defensive-strength pre-match rating; venue; red cards. Definitions follow CHOICE DATA-01.
- **Metric directions (higher/lower is better):** Higher clean-sheet probability and team defensive resistance are better; lower adjusted goals conceded and xGA are better.
- **Rate denominators (per 90/per appearance/per start/etc.):** Clean-sheet probability per eligible start; goals conceded and xGA per 90. A clean sheet counts only when the player meets FPL's minutes rule.
- **Context adjustments:** Adjust for opponent attack, team defence, venue, minutes, game state and red cards using pre-match information only.
- **Metric preprocessing/caps:** Cap continuous inputs at the 1st/99th positional percentiles; never treat matches below 60 minutes as clean-sheet opportunities; flag provider gaps.
- **Component standardization:** Position-relative z-scores for DEF and GKP separately; invert goals-conceded/xGA direction before combining.
- **Component weights or fitted method:** V1 fixed weights: 50% adjusted clean-sheet probability, 30% adjusted goals-conceded prevention, 20% team defensive-resistance index. Use these fixed weights under CHOICE FIT-01; change them only in a later version after walk-forward validation.
- **Raw score equation:** `0.50*z(adj_CS_probability) + 0.30*z(-adj_GC_or_xGA90) + 0.20*z(team_defensive_resistance)`.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Enter at the 80th percentile within FPL position unless an archetype-specific practical-effect rule is stricter.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Must be distinguished from team-level defensive-forecast signal, not player-family exclusivity
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict next-10-start clean-sheet probability and FPL clean-sheet points better than team clean-sheet rate alone; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### SAVES_MACHINE — Saves Machine

- **ID:** SAVES_MACHINE
- **Display name:** Saves Machine
- **Family:** Production
- **Definition in one sentence:** Generates FPL save points at a rate exceeding position-relative (goalkeeper) expectation.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** GKP only
- **Eligible player population:** Goalkeepers with sufficient minutes
- **Required raw fields and provider definitions:** minutes; saves; shots on target faced; post-shot expected goals if available; goals conceded; penalties saved; opponent attacking strength. Definitions follow CHOICE DATA-01.
- **Metric directions (higher/lower is better):** higher
- **Rate denominators (per 90/per appearance/per start/etc.):** Saves, shots faced and goals prevented per 90; save percentage per shot on target faced; penalty saves per penalty faced with strong shrinkage.
- **Context adjustments:** Adjust for opponent attack, team defensive resistance, venue, red cards, shots faced and minutes using pre-match/context information only.
- **Metric preprocessing/caps:** Require valid shots-on-target-faced denominators; winsorize continuous rates at the 1st/99th goalkeeper-season percentiles; shrink penalty rate heavily; flag missing post-shot xG.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** V1 fixed weights: 45% saves/90, 30% save rate above expectation, 15% shots-on-target faced/90, 10% penalty-save rate after strong shrinkage. If post-shot xG is unavailable, reassign its share to saves/90 and save percentage. Use these fixed weights under CHOICE FIT-01; change them only in a later version after walk-forward validation.
- **Raw score equation:** `0.45*z(saves90) + 0.30*z(goals_prevented90) + 0.15*z(SOT_faced90) + 0.10*z(shrunk_penalty_save_rate)`.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (GKP only) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Enter at the 80th percentile within FPL position unless an archetype-specific practical-effect rule is stricter.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** GKP-only production style, analogous to Shot Stopper in the goalkeeper production-style set
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict next-10-start save points and saves/90 better than saves/90 alone; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### Usage States — Nailed, Regular Starter, Rotation Risk, Impact Sub, Fringe

- **ID:** NAILED, REGULAR_STARTER, ROTATION_RISK, IMPACT_SUB, FRINGE
- **Display name:** Nailed, Regular Starter, Rotation Risk, Impact Sub, Fringe
- **Family:** Usage
- **Definition in one sentence:** A mutually exclusive state describing the player's current probability of starting and expected playing time when available.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** All rostered players with a defined recent appearance history
- **Required raw fields and provider definitions:** team match ID; squad availability; start flag; minutes; substitute appearance; expected minutes; injury/suspension reason code.
- **Metric directions (higher/lower is better):** Higher start probability and expected minutes imply a stronger/safer usage state.
- **Rate denominators (per 90/per appearance/per start/etc.):** Start and cameo probabilities per team match while available; minutes as average minutes per available match.
- **Context adjustments:** Condition on availability and separate injury/suspension absences from selection omissions; account for manager/club change.
- **Metric preprocessing/caps:** Minutes capped at the match maximum; double-gameweek fixtures remain separate matches; missing availability reason remains unknown rather than assumed injured.
- **Component standardization:** No position percentile is required for the state boundaries; use calibrated probabilities on 0–1 and expected minutes on 0–90.
- **Component weights or fitted method:** Calibrated start-probability and expected-minutes model; use cameo probability as the secondary splitter.
- **Raw score equation:** Model `P(start | available)`, `P(cameo | not starting, available)` and expected minutes separately; assign the mutually exclusive state from the thresholds.


- **Final score scale:** `usage_score = 100 × P(start | available)`. This is an absolute calibrated probability score, not a position-relative percentile. Return `expected_minutes` and `cameo_probability` separately.
- **Reference population:** All available Premier League players; position/club/squad-role groups may define priors and calibration strata, but the displayed usage score is not percentile-ranked.
- **Historical window:** Before the current season has a completed match, use every available team match from the immediately previous season with equal weight. Older seasons are excluded.
- **Earlier-current-season window:** All current-season team matches older than the latest 6 team matches.
- **Recent window:** Latest 6 team matches, including zero-minute selection omissions when the player was available.
- **Temporal combination formula:** Use equal weights across the complete previous season during preseason. After the current season starts, use the latest 6 team matches with exponential recency weighting and a 3-match half-life; reason-coded injury/suspension/illness rows receive half weight when they remain eligible for the calculation.
- **Minimum total minutes:** No hard minutes floor for a provisional state; require 450 available-match minutes or 8 availability observations for Medium/High confidence.
- **Minimum eligible appearances:** At least 3 available team-match observations for a provisional state and 8 for a full state.
- **Minimum starts (if relevant):** No universal start floor; start count contributes to confidence and calibrated start probability.
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Shrink start and cameo probabilities toward the position-by-club-squad-role prior, then toward the position mean if the role prior is sparse.
- **Label qualification threshold:** Apply the following mutually exclusive rules in precedence order.
    Nailed: start probability ≥0.90 and expected minutes ≥75. 
    Regular Starter: start probability ≥0.70 and expected minutes ≥55, unless Nailed. 
    Rotation Risk: start probability ≥0.35 or expected minutes ≥30, unless above. 
    Impact Sub: start probability <0.35 and cameo probability ≥0.35.
    Fringe: everyone else.

- **Exit threshold/hysteresis:** During the season, change state only after two consecutive distinct match-evidence samples propose the same new state. Re-publishing an unchanged sample must not advance the counter. The first preseason baseline, a usage-method version change, transfer or manager change applies the proposed state immediately and clears pending state.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Always return the most likely state, but mark it Insufficient Evidence below 0.35 and Provisional below 0.55; do not present it as a firm selection forecast.
- **Trend formula and thresholds:** Compare current start probability with the prior three-update average: Rising at +0.10, Declining at -0.10, Stable within ±0.05; require two updates.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Exactly one state. Resolve overlaps in this order using calibrated probabilities and expected minutes: Nailed, Regular Starter, Rotation Risk, Impact Sub, Fringe.
- **Transfer/new-club handling:** Reset club-specific usage evidence to a role prior and mark Provisional for 5 team matches; retain only broad player durability evidence at 25% weight.
- **Position-change handling:** Retain start/minutes evidence because usage is not position-relative, but rebuild the squad-role prior for the new listed position.
- **Injury/absence handling:** Exclude confirmed unavailable matches from start-rate denominators, but reduce forecast confidence; count unexplained available omissions at full weight and reason-coded injury/suspension omissions at half weight in recency.
- **Missing-data fallback:** If start or minutes are missing, return no usage state. If only availability reason is missing, treat it as unknown, lower data-quality confidence and do not infer injury.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version. Also persist start_probability, cameo_probability, expected_minutes, usage_evidence_fingerprint, usage_window_mode and usage_method_version.
- **Unit tests:** Test all state boundaries, no gaps/overlaps, injury exclusion, available omission treatment, transfer reset, preseason reset, unchanged-evidence publication, evidence fingerprint changes, expected-minute cap, double-gameweek separation and probability calibration.
- **Backtest target and acceptance criterion:** Predict next-match start and minutes with better Brier score/MAE than last-match and season-start-rate baselines; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### FODDER_HUNTER — Fodder Hunter

- **ID:** FODDER_HUNTER
- **Display name:** Fodder Hunter
- **Family:** Fixture
- **Definition in one sentence:** Shows a meaningful negative performance response as opponent strength increases.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient hard- and easy-fixture samples
- **Required raw fields and provider definitions:** eligible-match production response; continuous pre-match matchup difficulty; opponent attack/defence and team attack/defence; venue; minutes; role/start status; red cards. Response is xGI/90 for attackers and clean-sheet/save outcome for DEF/GKP.
- **Metric directions (higher/lower is better):** A more negative performance slope as fixture difficulty rises indicates stronger Fodder Hunter behaviour.
- **Rate denominators (per 90/per appearance/per start/etc.):** Use minutes as model exposure; an appearance enters the evidence set at 30+ minutes.
- **Context adjustments:** Control for team strength, venue, minutes, role and red cards; use only pre-match opponent ratings.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Hierarchical linear player slope, shrunk toward the FPL-position average; choose shrinkage strength by walk-forward testing.
- **Raw score equation:** Position percentile of the sign-reversed adjusted player difficulty slope. Easy-to-hard decline must also be at least 0.50 SD.


- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** At least 6 eligible appearances in each easy/medium/hard evidence band and opponents spanning at least 60% of the league-strength range; 8 per band for full confidence.
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Adjusted easy-to-hard performance decline of at least 0.50 SD, score at or above the 80th positional percentile, and confidence at least 0.70.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Fixture behaviour family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Same negative slope direction in at least two seasons/folds and improved held-out fixture-conditional production prediction; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### MATCHUP_PROOF — Matchup Proof

- **ID:** MATCHUP_PROOF
- **Display name:** Matchup Proof
- **Family:** Fixture
- **Definition in one sentence:** Shows near-zero performance response to opponent strength, given sufficient hard/easy fixture samples.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient hard- and easy-fixture samples
- **Required raw fields and provider definitions:** Same continuous matchup, response and control fields as Fodder Hunter.
- **Metric directions (higher/lower is better):** Closer to zero adjusted difficulty effect is better; this is an equivalence claim, not merely a non-significant slope.
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Hierarchical linear player slope shrunk toward the FPL-position average, with an explicit practical-equivalence test.
- **Raw score equation:** Score rises as the absolute adjusted easy-to-hard difference approaches zero; qualification requires the full effect interval to fit the equivalence rule.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** At least 6 eligible appearances in each easy/medium/hard evidence band and 60% opponent-strength coverage; 8 per band for full confidence.
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Adjusted hard-versus-easy difference within ±0.20 SD, score at or above the 80th positional percentile, and confidence at least 0.70.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`.

 Bands: 
 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight

Position-percentile scores: maximum_useful_width = 40 score points.
Probabilities: maximum_useful_width = 0.40.
Standardized fixture/venue effects: maximum_useful_width = 1.00 SD.
Start probability: maximum_useful_width = 0.40


- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Fixture behaviour family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Stable equivalence result in at least two seasons/folds and no meaningful fixture-interaction error out of sample; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### HOME_FAVORITE — Home Favorite

- **ID:** HOME_FAVORITE
- **Display name:** Home Favorite
- **Family:** Venue
- **Definition in one sentence:** Performs meaningfully better at home than away after adjustment.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient home and away sample sizes
- **Required raw fields and provider definitions:** eligible-match production response; home/away flag; opponent and team strength; expected minutes/start; role; red cards.
- **Metric directions (higher/lower is better):** Higher adjusted home-minus-away performance indicates stronger Home Favorite behaviour.
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Hierarchical adjusted home-effect estimate shrunk toward the FPL-position average.
- **Raw score equation:** Position percentile of adjusted `(home performance - away performance)`.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** At least 8 home and 8 away eligible appearances.
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Adjusted home advantage at least +0.35 SD, score at or above the 80th positional percentile, and confidence at least 0.70.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Venue behaviour family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Home advantage has the same direction in at least two folds and improves held-out venue-conditional prediction; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### ROAD_WARRIOR — Road Warrior

- **ID:** ROAD_WARRIOR
- **Display name:** Road Warrior
- **Family:** Venue
- **Definition in one sentence:** Performs meaningfully better away than at home after adjustment.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient home and away sample sizes
- **Required raw fields and provider definitions:** Same venue, response and control fields as Home Favorite.
- **Metric directions (higher/lower is better):** More negative adjusted home-minus-away performance indicates stronger Road Warrior behaviour.
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Hierarchical adjusted venue-effect estimate shrunk toward the FPL-position average.
- **Raw score equation:** Position percentile of adjusted `(away performance - home performance)`.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** At least 8 home and 8 away eligible appearances.
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Adjusted away advantage at least +0.35 SD, score at or above the 80th positional percentile, and confidence at least 0.70.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Venue behaviour family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Away advantage has the same direction in at least two folds and improves held-out venue-conditional prediction; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### ANYWHERE_THREAT — Anywhere Threat

- **ID:** ANYWHERE_THREAT
- **Display name:** Anywhere Threat
- **Family:** Venue
- **Definition in one sentence:** Performs equivalently home and away, within a defined equivalence band (not merely a failure to detect a difference).
- **V1 status:** KEEP V1. The earlier KEEP/CONSIDER conflict is resolved; the practical-equivalence band is ±0.20 SD.
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient home and away sample sizes
- **Required raw fields and provider definitions:** Same venue, response and control fields as Home Favorite.
- **Metric directions (higher/lower is better):** A smaller absolute adjusted home-away difference is better.
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Hierarchical venue-effect estimate plus a practical-equivalence test; shrink toward the FPL-position average.
- **Raw score equation:** Score rises as `abs(adjusted home - away performance)` approaches zero.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** At least 8 home and 8 away eligible appearances.
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Adjusted home-away difference within ±0.20 SD, score at or above the 80th positional percentile, and confidence at least 0.70.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Venue behaviour family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Venue equivalence is stable in at least two folds and held-out venue interaction remains practically negligible; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### EXPLOSIVE_RETURNER — Explosive Returner

- **ID:** EXPLOSIVE_RETURNER
- **Display name:** Explosive Returner
- **Family:** Return shape
- **Definition in one sentence:** Produces high-value ‘haul’ returns at an elevated rate relative to a defined threshold and denominator.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient appearances to establish a return-rate baseline
- **Required raw fields and provider definitions:** unmultiplied FPL points; minutes; eligible-appearance flag; haul event defined as 10+ FPL points.
- **Metric directions (higher/lower is better):** Higher shrunk probability of a 10+ point haul is better.
- **Rate denominators (per 90/per appearance/per start/etc.):** Haul probability per eligible appearance (start or 30+ minutes), not per 90.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Beta-binomial haul probability shrunk toward the FPL-position haul rate.
- **Raw score equation:** Posterior mean `P(FPL points >= 10 | eligible appearance)`, converted to a position percentile.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** Provisional: 15 eligible appearances and at least 2 hauls. Full evidence: 25 appearances and at least 3 hauls.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Score at or above the 80th positional percentile, adjusted haul probability at least 10%, and confidence at least 0.50.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Not mutually exclusive. A player may qualify as both Explosive and Steady because haul ceiling and return consistency measure different behaviour.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict held-out 10+ point haul probability with better Brier score/calibration than position-only and realized-points baselines; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### STEADY_RETURNER — Steady Returner

- **ID:** STEADY_RETURNER
- **Display name:** Steady Returner
- **Family:** Return shape
- **Definition in one sentence:** Maintains a high floor/return rate with low bust frequency; distinct from merely low variance.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Players with sufficient appearances to establish a return-rate baseline
- **Required raw fields and provider definitions:** unmultiplied FPL points; minutes; return event (goal, assist, clean sheet or save points as position-appropriate); blank event; appearance-level point variability; expected minutes.
- **Metric directions (higher/lower is better):** Higher return rate is better; lower blank rate and downside variability are better.
- **Rate denominators (per 90/per appearance/per start/etc.):** Return and blank rates per eligible appearance; production floor uses points per 90 and expected-minutes-adjusted points.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** V1 fixed weights: 50% return rate, 30% inverse blank rate, 20% inverse downside variability, after position standardization.
- **Raw score equation:** `0.50*z(return_rate) - 0.30*z(blank_rate) - 0.20*z(downside_variability)`.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 450 minutes for a provisional score; 900 minutes for an established label.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Score at or above the 80th positional percentile, production at or above the 60th positional percentile, and confidence at least 0.70.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Not mutually exclusive. A player may qualify as both Explosive and Steady because haul ceiling and return consistency measure different behaviour.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict held-out return frequency and low-downside consistency better than mean FPL points alone; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### HIDDEN_GEM — Hidden Gem

- **ID:** HIDDEN_GEM
- **Display name:** Hidden Gem
- **Family:** Value
- **Definition in one sentence:** Low-cost player producing strong position-relative value.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Active Premier League players with a current FPL price and replacement-level comparison; a full badge requires the stated historical or forward playing-time floor.
- **Required raw fields and provider definitions:** start-of-gameweek and current price; FPL position; realized points; expected next-five-GW points; expected minutes; availability; replacement-player curve.
- **Metric directions (higher/lower is better):** The label represents cheap and high production/value; production/value direction follows that definition.
- **Rate denominators (per 90/per appearance/per start/etc.):** Historical view uses realized points per million and points over replacement; forward view uses expected next-five-GW points over replacement. Expected minutes are applied once only.
- **Context adjustments:** Forward view adjusts expected points for fixtures, team strength, venue, role and expected minutes; historical view is descriptive and remains separate under CHOICE VALUE-01.
- **Metric preprocessing/caps:** Use start-of-GW price for historical snapshots and current price for forecasts; no retroactive repricing; cap extreme value residuals at the 1st/99th position percentiles.
- **Component standardization:** Price and production/value percentiles calculated separately within FPL position.
- **Component weights or fitted method:** Two-axis rule, not a weighted average: price band plus value-over-replacement band.
- **Raw score equation:** Historical: realized points above position replacement at the recorded price. Forward: expected next-five-GW points above the best regular player near the position base price. Display views separately per CHOICE VALUE-01.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** TIME-01 applies to the historical production inputs used by the expected-points model. Historical delivered value uses the resulting historical production estimate. Forward value is calculated directly from current price and expected next-five-GW points over replacement; do not temporally blend the final forward value score again.
- **Minimum total minutes:** At least 450 historical minutes and 50 expected minutes for the forward horizon; otherwise score only and mark Provisional.
- **Minimum eligible appearances:** At least 10 eligible historical appearances for a full historical value badge; forward view may use a prior but remains Provisional.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** bottom 25% of position price and at or above the 80th percentile for value-over-replacement; use entry at 80/20 and exit at 75/25 as appropriate.
- **Exit threshold/hysteresis:** For a cheap label, retain until price rises above the 30th position percentile; for high value retain until value falls below the 75th percentile. Require two updates.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Value state family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Evaluate future five-GW points over replacement, rank calibration and label stability versus simple points-per-million; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### PREMIUM_PICK — Premium Pick

- **ID:** PREMIUM_PICK
- **Display name:** Premium Pick
- **Family:** Value
- **Definition in one sentence:** High-cost player producing elite production commensurate with price.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Active Premier League players with a current FPL price and replacement-level comparison; a full badge requires the stated historical or forward playing-time floor.
- **Required raw fields and provider definitions:** start-of-gameweek and current price; FPL position; realized points; expected next-five-GW points; expected minutes; availability; replacement-player curve.
- **Metric directions (higher/lower is better):** The label represents expensive and high production/value; production/value direction follows that definition.
- **Rate denominators (per 90/per appearance/per start/etc.):** Historical view uses realized points per million and points over replacement; forward view uses expected next-five-GW points over replacement. Expected minutes are applied once only.
- **Context adjustments:** Forward view adjusts expected points for fixtures, team strength, venue, role and expected minutes; historical view is descriptive and remains separate under CHOICE VALUE-01.
- **Metric preprocessing/caps:** Use start-of-GW price for historical snapshots and current price for forecasts; no retroactive repricing; cap extreme value residuals at the 1st/99th position percentiles.
- **Component standardization:** Price and production/value percentiles calculated separately within FPL position.
- **Component weights or fitted method:** Two-axis rule, not a weighted average: price band plus value-over-replacement band.
- **Raw score equation:** Historical: realized points above position replacement at the recorded price. Forward: expected next-five-GW points above the best regular player near the position base price. Display views separately per CHOICE VALUE-01.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** TIME-01 applies to the historical production inputs used by the expected-points model. Historical delivered value uses the resulting historical production estimate. Forward value is calculated directly from current price and expected next-five-GW points over replacement; do not temporally blend the final forward value score again.
- **Minimum total minutes:** At least 450 historical minutes and 50 expected minutes for the forward horizon; otherwise score only and mark Provisional.
- **Minimum eligible appearances:** At least 10 eligible historical appearances for a full historical value badge; forward view may use a prior but remains Provisional.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** top 25% of position price and at or above the 80th percentile for value-over-replacement; use entry at 80/20 and exit at 75/25 as appropriate.
- **Exit threshold/hysteresis:** For a premium label, retain until price falls below the 70th position percentile; for high value retain until value falls below the 75th percentile. Require two updates.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Value state family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Evaluate future five-GW points over replacement, rank calibration and label stability versus simple points-per-million; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### HIGH_MAINTENANCE — High Maintenance

- **ID:** HIGH_MAINTENANCE
- **Display name:** High Maintenance
- **Family:** Value
- **Definition in one sentence:** High-cost player producing weak or mediocre production relative to price.
- **V1 status:** KEEP V1
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Active Premier League players with a current FPL price and replacement-level comparison; a full badge requires the stated historical or forward playing-time floor.
- **Required raw fields and provider definitions:** start-of-gameweek and current price; FPL position; realized points; expected next-five-GW points; expected minutes; availability; replacement-player curve.
- **Metric directions (higher/lower is better):** The label represents expensive and low production/value; production/value direction follows that definition.
- **Rate denominators (per 90/per appearance/per start/etc.):** Historical view uses realized points per million and points over replacement; forward view uses expected next-five-GW points over replacement. Expected minutes are applied once only.
- **Context adjustments:** Forward view adjusts expected points for fixtures, team strength, venue, role and expected minutes; historical view is descriptive and remains separate under CHOICE VALUE-01.
- **Metric preprocessing/caps:** Use start-of-GW price for historical snapshots and current price for forecasts; no retroactive repricing; cap extreme value residuals at the 1st/99th position percentiles.
- **Component standardization:** Price and production/value percentiles calculated separately within FPL position.
- **Component weights or fitted method:** Two-axis rule, not a weighted average: price band plus value-over-replacement band.
- **Raw score equation:** Historical: realized points above position replacement at the recorded price. Forward: expected next-five-GW points above the best regular player near the position base price. Display views separately per CHOICE VALUE-01.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** TIME-01 applies to the historical production inputs used by the expected-points model. Historical delivered value uses the resulting historical production estimate. Forward value is calculated directly from current price and expected next-five-GW points over replacement; do not temporally blend the final forward value score again.
- **Minimum total minutes:** At least 450 historical minutes and 50 expected minutes for the forward horizon; otherwise score only and mark Provisional.
- **Minimum eligible appearances:** At least 10 eligible historical appearances for a full historical value badge; forward view may use a prior but remains Provisional.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** top 25% of position price and at or below the 20th percentile for value-over-replacement; use entry at 80/20 and exit at 75/25 as appropriate.
- **Exit threshold/hysteresis:** For a premium label, retain until price falls below the 70th position percentile; for low value retain until value rises above the 25th percentile. Require two updates.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Value state family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Evaluate future five-GW points over replacement, rank calibration and label stability versus simple points-per-million; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### BENCH_AUTOFILL — Bench Auto-fill

- **ID:** BENCH_AUTOFILL
- **Display name:** Bench Auto-fill
- **Family:** Value
- **Definition in one sentence:** Low-cost player with low production, used primarily as squad filler.
- **V1 status:** KEEP V1. For Bench Auto-fill, the earlier KEEP/CONSIDER conflict is resolved in favor of KEEP with the same evidence floor as other value labels.
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** Active Premier League players with a current FPL price and replacement-level comparison; a full badge requires the stated historical or forward playing-time floor.
- **Required raw fields and provider definitions:** start-of-gameweek and current price; FPL position; realized points; expected next-five-GW points; expected minutes; availability; replacement-player curve.
- **Metric directions (higher/lower is better):** The label represents cheap and low production/value; production/value direction follows that definition.
- **Rate denominators (per 90/per appearance/per start/etc.):** Historical view uses realized points per million and points over replacement; forward view uses expected next-five-GW points over replacement. Expected minutes are applied once only.
- **Context adjustments:** Forward view adjusts expected points for fixtures, team strength, venue, role and expected minutes; historical view is descriptive and remains separate under CHOICE VALUE-01.
- **Metric preprocessing/caps:** Use start-of-GW price for historical snapshots and current price for forecasts; no retroactive repricing; cap extreme value residuals at the 1st/99th position percentiles.
- **Component standardization:** Price and production/value percentiles calculated separately within FPL position.
- **Component weights or fitted method:** Two-axis rule, not a weighted average: price band plus value-over-replacement band.
- **Raw score equation:** Historical: realized points above position replacement at the recorded price. Forward: expected next-five-GW points above the best regular player near the position base price. Display views separately per CHOICE VALUE-01.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** TIME-01 applies to the historical production inputs used by the expected-points model. Historical delivered value uses the resulting historical production estimate. Forward value is calculated directly from current price and expected next-five-GW points over replacement; do not temporally blend the final forward value score again.
- **Minimum total minutes:** At least 450 historical minutes and 50 expected minutes for the forward horizon; otherwise score only and mark Provisional.
- **Minimum eligible appearances:** At least 10 eligible historical appearances for a full historical value badge; forward view may use a prior but remains Provisional.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** bottom 25% of position price and at or below the 20th percentile for value-over-replacement; use entry at 80/20 and exit at 75/25 as appropriate.
- **Exit threshold/hysteresis:** For a cheap label, retain until price rises above the 30th position percentile; for low value retain until value rises above the 25th percentile. Require two updates.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Value state family is exclusive.
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Evaluate future five-GW points over replacement, rank calibration and label stability versus simple points-per-million; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### POINTS_HAZARD — Points Hazard

- **ID:** POINTS_HAZARD
- **Display name:** Points Hazard
- **Family:** Risk badge
- **Definition in one sentence:** Elevated risk of point-losing disciplinary events (e.g. cards) relative to position-relative baseline.
- **V1 status:** KEEP V1 (secondary badge, not a primary archetype)
- **Eligible FPL positions:** FWD, MID, DEF, GKP
- **Eligible player population:** All rostered players with sufficient appearances
- **Required raw fields and provider definitions:** minutes; yellow cards; second-yellow dismissals; straight red cards; current card total; competition suspension thresholds; active suspension flag.
- **Metric directions (higher/lower is better):** Higher expected FPL point loss and suspension risk mean stronger hazard.
- **Rate denominators (per 90/per appearance/per start/etc.):** Continuous production metrics per 90; event probabilities per eligible appearance/start, as stated for the archetype.
- **Context adjustments:** Adjust for pre-match team strength, opponent strength, venue, minutes/exposure, role and red cards where relevant; never use post-match or future information.
- **Metric preprocessing/caps:** Validate units and impossible values; winsorize continuous rates at the 1st/99th position-season percentiles; apply log1p to strongly skewed count rates; retain explicit missing flags.
- **Component standardization:** Convert components to z-scores within FPL position and season using the full eligible 20-team reference population, then convert the final shrunk score to a 0–100 percentile.
- **Component weights or fitted method:** Expected FPL point loss per 90 from card events plus near-term suspension probability; shrink event rates toward the FPL-position mean.
- **Raw score equation:** `expected_card_deduction90 + expected_suspension_minutes_lost90`, standardized within position.
- **Final score scale:** 0–100 position-relative percentile; 100 is the strongest expression of the archetype.
- **Reference population:** Players within the position group(s) named under Eligible FPL positions (FWD, MID, DEF, GKP) who also satisfy Eligible player population; archetype score is computed position-relative within this group, not against the full player pool.
- **Historical window:** The most recent completed Premier League season; older seasons are excluded in V1 unless needed for a promoted/new player prior.
- **Earlier-current-season window:** All current-season eligible appearances older than the latest 10.
- **Recent window:** Latest 10 eligible appearances; an eligible appearance is a start or 30+ minutes unless the archetype states otherwise.
- **Temporal combination formula:** Apply TIME-01: combine previous-season, earlier-current-season, and recent scores using base weights 25%/30%/45%. If a window is unavailable, omit it and renormalize the remaining weights using the TIME-01 equation. Apply small-sample shrinkage separately after temporal combination.
- **Minimum total minutes:** 900 minutes for a full label; 450–899 minutes may show a provisional score only.
- **Minimum eligible appearances:** 10 eligible appearances for a label unless a stricter archetype-specific rule is stated.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Empirical-Bayes/pseudo-count shrinkage toward the current FPL-position mean, with shrinkage weakening as eligible minutes and events increase. Select the prior strength through walk-forward testing, using a predefined search grid such as 450, 900, 1,350 and 1,800 equivalent minutes.
- **Label qualification threshold:** Top 15% of position-relative discipline risk and confidence at least 0.70. Active suspension is displayed separately, not inferred by the score.
- **Exit threshold/hysteresis:** Retain until the score falls below the 75th positional percentile for two consecutive updates, unless an archetype-specific state boundary applies.
- **Confidence formula and display bands:** `0.30*minutes_adequacy + 0.15*appearance_adequacy + 0.20*context_coverage + 0.20*precision + 0.15*data_quality`. Bands: 
    Insufficient <0.35; 
    Low 0.35–0.54; 
    Medium 0.55–0.74; 
    High ≥0.75.
    minutes_adequacy = min(eligible_minutes / required_full_minutes, 1)
    appearance_adequacy = min(eligible_appearances / required_appearances, 1)
    context_coverage = minimum proportion of required context samples met
    precision = max(0, min(1, 1 - interval_width / maximum_useful_width))
    data_quality = available planned component weight / total planned weight
- **Suppression rule at low confidence:** Store the score but hide the active label below 0.35 confidence; show a Provisional label from 0.35 until the archetype's full confidence/evidence requirement is met.
- **Trend formula and thresholds:** Recent score minus historical/earlier-current baseline: Rising at +10 or more score points, Declining at -10 or less, Stable within ±5; require the direction for two consecutive updates. Values between 5 and 10 are Emerging/soft trend only.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Badges family is not mutually exclusive; can co-occur with any primary archetype
- **Transfer/new-club handling:** Carry individual-event evidence forward, but shrink club/context-adjusted effects 50% toward the new position mean for the first 5 eligible appearances at the new club; rebuild team-context inputs immediately from the new club.
- **Position-change handling:** Keep raw match evidence, recompute every standardized score against the new FPL position, and show Provisional until 5 eligible appearances or 450 minutes in the new position/role.
- **Injury/absence handling:** Do not lower production ability solely because of injury, suspension or absence. Age evidence normally and lower confidence after 30 days without an eligible appearance. Usage state uses a separate short recency half-life and reason-coded absences receive half weight.
- **Missing-data fallback:** If a core metric is missing, return no label and record the reason. If only secondary metrics are missing, reweight available components only when at least 70% of planned weight remains, reduce data-quality confidence proportionally, and never fail silently.
- **Output fields:** player_id, snapshot_date, archetype_id, display_name, family, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, evidence_minutes, eligible_appearances, component_scores, missing_data_flags, model_version.
- **Unit tests:** Test threshold entry/exit, position-relative ranking, minimum evidence suppression, shrinkage of tiny samples, no future-data leakage, missing-core-field suppression, secondary-field reweighting, transfer/position reset, injury non-penalization, and score bounds.
- **Backtest target and acceptance criterion:** Predict held-out card deductions and suspension-caused minutes lost better than cards/90 alone; release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

---

## Part 2 — Production Style position combinations (28 entries)

### FWD

#### PRODSTYLE_FWD_G — Finisher

- **ID:** PRODSTYLE_FWD_G
- **Display name:** Finisher
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the G production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_FWD_C — Creative Forward

- **ID:** PRODSTYLE_FWD_C
- **Display name:** Creative Forward
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the C production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Creator.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_FWD_D — Pressing Forward

- **ID:** PRODSTYLE_FWD_D
- **Display name:** Pressing Forward
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_FWD_G_C — Complete Forward

- **ID:** PRODSTYLE_FWD_G_C
- **Display name:** Complete Forward
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the G+C production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Creator.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_FWD_G_D — Pressing Finisher

- **ID:** PRODSTYLE_FWD_G_D
- **Display name:** Pressing Finisher
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the G+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_FWD_C_D — Pressing Link Forward

- **ID:** PRODSTYLE_FWD_C_D
- **Display name:** Pressing Link Forward
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the C+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Creator, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_FWD_G_C_D — Complete Two-Way Forward

- **ID:** PRODSTYLE_FWD_G_C_D
- **Display name:** Complete Two-Way Forward
- **Family:** Production style
- **Definition in one sentence:** A FWD player who simultaneously qualifies for the G+C+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** FWD
- **Eligible player population:** FWD players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Creator, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### MID

#### PRODSTYLE_MID_G — Goal-Scoring Midfielder

- **ID:** PRODSTYLE_MID_G
- **Display name:** Goal-Scoring Midfielder
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the G production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_MID_C — Playmaker

- **ID:** PRODSTYLE_MID_C
- **Display name:** Playmaker
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the C production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Creator.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_MID_D — Ball-Winning Midfielder

- **ID:** PRODSTYLE_MID_D
- **Display name:** Ball-Winning Midfielder
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_MID_G_C — Attacking Playmaker

- **ID:** PRODSTYLE_MID_G_C
- **Display name:** Attacking Playmaker
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the G+C production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Creator.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_MID_G_D — Box-to-Box Midfielder

- **ID:** PRODSTYLE_MID_G_D
- **Display name:** Box-to-Box Midfielder
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the G+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_MID_C_D — Deep-Lying Playmaker

- **ID:** PRODSTYLE_MID_C_D
- **Display name:** Deep-Lying Playmaker
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the C+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Creator, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_MID_G_C_D — Complete Midfielder

- **ID:** PRODSTYLE_MID_G_C_D
- **Display name:** Complete Midfielder
- **Family:** Production style
- **Definition in one sentence:** A MID player who simultaneously qualifies for the G+C+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** MID
- **Eligible player population:** MID players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Creator, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### DEF

#### PRODSTYLE_DEF_G — Goal-Threat Defender

- **ID:** PRODSTYLE_DEF_G
- **Display name:** Goal-Threat Defender
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the G production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_DEF_C — Creative Defender

- **ID:** PRODSTYLE_DEF_C
- **Display name:** Creative Defender
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the C production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Creator.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_DEF_D — Defensive Stopper

- **ID:** PRODSTYLE_DEF_D
- **Display name:** Defensive Stopper
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_DEF_G_C — Attacking Defender

- **ID:** PRODSTYLE_DEF_G_C
- **Display name:** Attacking Defender
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the G+C production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Creator.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_DEF_G_D — Two-Way Defender

- **ID:** PRODSTYLE_DEF_G_D
- **Display name:** Two-Way Defender
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the G+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_DEF_C_D — Defensive Creator

- **ID:** PRODSTYLE_DEF_C_D
- **Display name:** Defensive Creator
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the C+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Creator, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_DEF_G_C_D — Complete Defender

- **ID:** PRODSTYLE_DEF_G_C_D
- **Display name:** Complete Defender
- **Family:** Production style
- **Definition in one sentence:** A DEF player who simultaneously qualifies for the G+C+D production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** DEF
- **Eligible player population:** DEF players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Goal Threat, Creator, Defensive Engine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

### GKP

#### PRODSTYLE_GKP_S — Shot Stopper

- **ID:** PRODSTYLE_GKP_S
- **Display name:** Shot Stopper
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the S production components.
- **V1 status:** KEEP V1 (component archetypes Goal Threat/Creator/Defensive Engine are KEEP V1; this row is the position-specific composite display name)
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Saves Machine.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_GKP_W — Sweeper Keeper

- **ID:** PRODSTYLE_GKP_W
- **Display name:** Sweeper Keeper
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the W production components.
- **V1 status:** DEFER V1 — requires the deferred Sweeper and/or Distributor base component.
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Sweeper Keeper base component. The W/P base definitions are outside V1.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold. This composite is not evaluated in V1 and can only be enabled after every required base component has a released specification.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_GKP_P — Distributor

- **ID:** PRODSTYLE_GKP_P
- **Display name:** Distributor
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the P production components.
- **V1 status:** DEFER V1 — requires the deferred Sweeper and/or Distributor base component.
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Distributor base component. The W/P base definitions are outside V1.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold. This composite is not evaluated in V1 and can only be enabled after every required base component has a released specification.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_GKP_S_W — Proactive Shot Stopper

- **ID:** PRODSTYLE_GKP_S_W
- **Display name:** Proactive Shot Stopper
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the S+W production components.
- **V1 status:** DEFER V1 — requires the deferred Sweeper and/or Distributor base component.
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Saves Machine, Sweeper Keeper base component. The W/P base definitions are outside V1.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold. This composite is not evaluated in V1 and can only be enabled after every required base component has a released specification.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_GKP_S_P — Ball-Playing Shot Stopper

- **ID:** PRODSTYLE_GKP_S_P
- **Display name:** Ball-Playing Shot Stopper
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the S+P production components.
- **V1 status:** DEFER V1 — requires the deferred Sweeper and/or Distributor base component.
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Saves Machine, Distributor base component. The W/P base definitions are outside V1.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold. This composite is not evaluated in V1 and can only be enabled after every required base component has a released specification.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_GKP_W_P — Ball-Playing Sweeper

- **ID:** PRODSTYLE_GKP_W_P
- **Display name:** Ball-Playing Sweeper
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the W+P production components.
- **V1 status:** DEFER V1 — requires the deferred Sweeper and/or Distributor base component.
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Sweeper Keeper base component, Distributor base component. The W/P base definitions are outside V1.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold. This composite is not evaluated in V1 and can only be enabled after every required base component has a released specification.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.

#### PRODSTYLE_GKP_S_W_P — Complete Goalkeeper

- **ID:** PRODSTYLE_GKP_S_W_P
- **Display name:** Complete Goalkeeper
- **Family:** Production style
- **Definition in one sentence:** A GKP player who simultaneously qualifies for the S+W+P production components.
- **V1 status:** DEFER V1 — requires the deferred Sweeper and/or Distributor base component.
- **Eligible FPL positions:** GKP
- **Eligible player population:** GKP players with sufficient minutes in each contributing component
- **Required raw fields and provider definitions:** Inherit the union of the required raw fields from: Saves Machine, Sweeper Keeper base component, Distributor base component. The W/P base definitions are outside V1.
- **Metric directions (higher/lower is better):** Higher component scores mean stronger qualification; every required component must be active.
- **Rate denominators (per 90/per appearance/per start/etc.):** Inherit each base component's denominator; do not recompute a second rate for the composite.
- **Context adjustments:** Inherit the adjustments from each base component.
- **Metric preprocessing/caps:** Inherit preprocessing and caps from each base component.
- **Component standardization:** Use the already standardized 0–100 component scores; do not standardize them a second time.
- **Component weights or fitted method:** No new fitted weights. The composite is a deterministic combination of active base labels.
- **Raw score equation:** Composite score = minimum of the required active component scores; this prevents one very high component from hiding a weak required component.
- **Final score scale:** 0–100; equal to the minimum score among the required base components.
- **Reference population:** Inherited from the player's current FPL position and each required base component.
- **Historical window:** Inherit each base component's historical window.
- **Earlier-current-season window:** Inherit each base component's earlier-current-season window.
- **Recent window:** Inherit each base component's recent window.
- **Temporal combination formula:** No separate temporal blend; inherit the current temporally blended score from each base component.
- **Minimum total minutes:** Inherit the strictest minimum among the required base components.
- **Minimum eligible appearances:** Inherit the strictest minimum among the required base components.
- **Minimum starts (if relevant):** N/A — not a start-dependent archetype
- **Minimum context samples (if relevant):** N/A — archetype is not defined by a specific context split
- **Small-sample shrinkage:** Applied only in the base components; composite confidence uses the weakest component.
- **Label qualification threshold:** Activate only when every named base component is active above its entry threshold. This composite is not evaluated in V1 and can only be enabled after every required base component has a released specification.
- **Exit threshold/hysteresis:** Retain only while every required base component remains above its own exit threshold.
- **Confidence formula and display bands:** Composite confidence = minimum confidence of the required base components; use the scheme-wide display bands.
- **Suppression rule at low confidence:** Suppress when any required base label is suppressed or unavailable.
- **Trend formula and thresholds:** Composite trend follows the minimum required component's score change; also return individual component trends.
- **Allowed statuses:** Insufficient Evidence, Provisional, Emerging, Established, Stable, Declining.
- **Within-family exclusivity/precedence:** Base components are non-exclusive internally, but the user-facing Production Style composite is exclusive. Display exactly one label using the global most-specific rule: three-component > two-component > single-component. Retain all base component scores internally and ignore deferred components.
- **Transfer/new-club handling:** Inherit base-component transfer handling and mark the composite Provisional if any required component is Provisional.
- **Position-change handling:** Recompute all base components in the new FPL-position reference group; map to the new position's display-name table.
- **Injury/absence handling:** Inherit base-component absence handling; absence lowers confidence, not the composite ability score directly.
- **Missing-data fallback:** Suppress the composite if any required base component cannot be calculated; never infer a missing component.
- **Output fields:** player_id, snapshot_date, composite_id, display_name, score_0_100, active_label, confidence_0_1, confidence_band, trend, status, required_component_ids, component_scores, component_confidences, missing_data_flags, model_version.
- **Unit tests:** Test exact component activation, weakest-score calculation, weakest-confidence calculation, exit behavior, missing-component suppression and position remapping.
- **Backtest target and acceptance criterion:** Validate deterministic mapping, stability and incremental usefulness of the component vector; the composite does not need to beat its own components as a separate predictive model. Release bar follows CHOICE VALID-01.
- **Owner/version/effective date:** Owner: Okechi Ukandu; specification version 1.0; effective 2026-08-17.


