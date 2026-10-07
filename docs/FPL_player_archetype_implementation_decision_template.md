# FPL Player Archetype System — Implementation Decision Workbook

> Status: supporting decision workbook, despite the historical ?template?
> filename. The [V1 catalogue](fpl_archetype_scheme_catalogue.md) governs model
> definitions; [V1 operations](FPL_ARCHETYPE_V1.md) describes the implementation.
> Unresolved sign-off placeholders below are not evidence of release approval.

**Purpose:** This document now contains the user's selections plus recommended V1 defaults for every previously open choice.

**Status notation**

- `AGREED`: already established in the current specification or prior discussion.
- `RECOMMENDED`: proposed default; accept or replace it explicitly.
- `DEFERRED`: deliberately excluded from V1 unless promoted.
- `ILLUSTRATIVE`: example only; not an agreed parameter.

Do not answer a field with words such as “high,” “recent,” “enough,” or “significant” unless the document also gives a numerical definition and unit.

---

## 1. Current agreed architecture

### 1.1 Purpose and output

`AGREED` The archetype system is a historical behavioural and explanation layer. It describes how a player has generated FPL value, under what conditions, and with what usage/return shape. It is not itself the final points-prediction model.

`AGREED` A player has a multi-dimensional profile rather than one exclusive global archetype. Labels can coexist across families; within-family exclusivity must be specified family by family.

`AGREED` Each scored archetype should expose:

$$
\boxed{\text{Score},\ \text{Confidence},\ \text{Trend/Status}}
$$

`AGREED` Calculate continuous scores first. Human-readable labels are thresholded interpretations of those scores.

`AGREED` Player comparisons are position-relative unless a particular specification explicitly overrides that rule.

### 1.2 Player evidence lifecycle

At season start:

$$
\text{previous-season evidence}\rightarrow\text{starting player profile}
$$

During the season:

$$
\text{previous season}+\text{earlier current season}+\text{recent current season}
\rightarrow\text{current profile}
$$

`AGREED` The recent window is the player's latest **10 eligible appearances**. What makes an appearance eligible remains to be defined. Older current-season evidence is retained rather than discarded.

`AGREED` In principle:

$$
\text{recent evidence}>\text{earlier-current evidence}>\text{previous-season evidence}
$$

but update speed and weights may differ by family. Usage should generally react faster than venue or fixture behaviour.

### 1.3 Team context

`AGREED` Keep three related but distinct team concepts:

1. Overall Elo: common general-quality prior/anchor.
2. Attacking strength: ability to create/convert attacking opportunities.
3. Defensive strength (or defensive resistance): ability to prevent opposition opportunities.

`AGREED` Prefer the hierarchical-prior design: overall Elo initializes or shrinks specialist attack/defence ratings; fixture predictions use specialist ratings. Do not mechanically add the same Elo information twice.

For an attacking fixture:

$$
\eta_{i,j,t}=\mu+h_{i,t}+A_{i,t}-D_{j,t}
$$

For defensive FPL potential, reverse the interaction: team defence versus opponent attack.

`AGREED` All ratings used for a match must be pre-match values calculated only from information available before that match. Validation is chronological/walk-forward, never a random split.

`AGREED` Established clubs use previous-Premier-League-season evidence. Promoted clubs receive an explicit conservative prior. All 20 clubs must exist before final league-wide standardization.

### 1.4 System boundaries

**Selected use:** Both descriptive UI and predictive-model features in V1.

If used as model features, use only out-of-fold archetype scores, never thresholded labels, and require incremental value beyond the raw inputs.

---

## 2. V1 scope freeze

For every item, replace the proposed status with one of `KEEP V1`, `CONSIDER V1`, `DEFER`, or `REMOVE`.

### 2.1 Recommended V1 core

| Family | Item | Proposed status | Final status | Final display name |
|---|---|---|---|---|
| Production | Goal Threat | KEEP V1 | `KEEP V1` | `Goal Threat` |
| Production | Creator | KEEP V1 | `KEEP V1` | `Creator` |
| Production | Defensive Engine | KEEP V1 | `KEEP V1` | `Defensive Engine` |
| Production | All-Rounder | KEEP V1 | `KEEP V1` | `Jack of All Trades` |
| Production | Clean-Sheet Dependent / Specialist | KEEP V1 | `KEEP V1` | `Clean-Sheet Specialist` |
| Production | Save Machine | KEEP V1 | `KEEP V1` | `Saves Machine` |
| Usage | Nailed | KEEP V1 | `KEEP V1` | `Undroppable` |
| Usage | Regular Starter / Starter | KEEP V1 | `KEEP V1` | `Regular Starter` |
| Usage | Rotation Risk | KEEP V1 | `KEEP V1` | `Rotation Risk` |
| Usage | Impact Sub | KEEP V1 | `KEEP V1` | `Impact Sub` |
| Usage | Fringe | KEEP V1 | `KEEP V1` | `Benchwarmer` |
| Fixture | Fodder Hunter | KEEP V1 | `KEEP V1` | `Fodder Hunter` |
| Fixture | Fixture Proof | KEEP V1 | `KEEP V1` | `Matchup Proof` |
| Venue | Home Comfort | KEEP V1 | `KEEP V1` | `Home Favorite` |
| Venue | Road Warrior | KEEP V1 | `KEEP V1` | `Road Warrior` |
| Venue | Venue Neutral / Venue Proof | KEEP or CONSIDER | `KEEP V1` | `Anywhere Threat` |
| Return shape | Explosive / Haul Threat | KEEP V1 | `KEEP V1` | `Explosive Returner` |
| Return shape | Reliable / Steady Returner | KEEP V1 | `KEEP V1` | `Steady Returner` |
| Value | Sleeper Agent | KEEP V1 | `KEEP V1` | `Hidden Gem` |
| Value | Worth the Hype / Premium Performer | KEEP V1 | `KEEP V1` | `Premium Pick` |
| Value | High Maintenance | KEEP V1 | `KEEP V1` | `High Maitenance` |
| Value | Budget Fodder | KEEP or CONSIDER | `KEEP V1` | `Bench Auto-fill` |
| Risk badge | Points Hazard | KEEP V1 | `KEEP V1` | Points Hazard |

### 2.2 Complete discussed catalogue

This is the complete consolidated set from the discussion, including renamed concepts and ideas that should not automatically enter V1.

| Family | Canonical concept | Names/aliases discussed | Proposed disposition | Reason or unresolved issue |
|---|---|---|---|---|
| Production | Goal Threat | Goal Scorer; goal-scoring midfielder | KEEP V1 | Use process plus outcomes; position-relative; not MID-only. |
| Production | Creator | Creative midfielder | KEEP V1 | Use xA/chance creation, not assists alone. |
| Production | Defensive Engine | Ground-Eating Midfielder; defensive midfielder | KEEP V1 | Position/data-source eligibility must be defined. |
| Production | Clean-Sheet Dependent | Clean-Sheet Specialist; CS Defender | KEEP V1 | Must distinguish points source from team defensive forecast. |
| Production | Save Machine | Superman | KEEP V1 | Goalkeepers only. |
| Production | All-Rounder | — | CONSIDER V1 | Requires breadth and minimum-component rules. |
| Production | Bonus Magnet | — | CONSIDER/DEFER | Bonus may duplicate other production and BPS rules. |
| Usage | Nailed | Nailed Starter | KEEP V1 | Separate start security from 90-minute completion. |
| Usage | Regular Starter | Starter | KEEP V1 | Needs exact start/minutes bands. |
| Usage | Rotation Risk | Rotation Player | KEEP V1 | Needs mixture-of-starts rule. |
| Usage | Impact Sub | Bench Player (partial replacement) | KEEP V1 | Low starts plus meaningful cameos. |
| Usage | Fringe | Bench Player (very low usage) | KEEP V1 | Low overall expected/observed minutes. |
| Usage | 90-Minute Man | — | REMOVE | Distinct from merely starting. |
| Fixture | Fodder Hunter | — | KEEP V1 | Negative response to opponent strength. |
| Fixture | Fixture Proof | — | KEEP V1 | Near-zero response with sufficient hard/easy samples. |
| Fixture | Big-Game Performer | — | DEFER | Positive difficult-fixture effect is likely noisy. |
| Venue | Home Comfort | Home Is Where the Heart Is; Home Performer/Specialist | KEEP V1 | Requires adjusted home-versus-away effect. |
| Venue | Road Warrior | Away Performer; Take Him Anywhere | KEEP V1 | “Take Him Anywhere” was also used for venue proof; choose one meaning. |
| Venue | Venue Neutral | Venue Proof; Take Him Anywhere | KEEP/CONSIDER | Define equivalence band, not failure to reject difference. |
| Return shape | Explosive | Haul Threat | KEEP V1 | Define haul threshold and denominator. |
| Return shape | Reliable | Mr Reliable; Steady Eddie; Steady Returner | KEEP V1 | High floor/return rate is not just low variance. |
| Return shape | Streaky | — | DEFER | Must show temporal clustering after fixture/minutes controls. |
| Return shape | Boom-or-Bust | — | DEFER | High haul rate plus frequent blanks; overlaps Explosive. |
| Value | Sleeper Agent | Sleeper; Bargain | KEEP V1 | Cheap plus strong position-relative value. |
| Value | Worth the Hype | Premium Performer | KEEP V1 | Expensive plus elite production. |
| Value | High Maintenance | Overpriced | KEEP V1 | Expensive plus weak/mediocre production. |
| Value | Budget Fodder | — | KEEP/CONSIDER | Cheap plus low production; clarify whether minimum minutes matter. |
| Risk | Points Hazard | Discipline Risk | KEEP V1 BADGE | Secondary badge, not primary archetype. |
| Risk | Clean Record | — | REMOVE | Optional inverse discipline badge. |
| Ranking | Top Points Provider | Top-two points provider | REMOVE AS ARCHETYPE | Keep only as rank/statistic; it does not describe behaviour. |

Are labels within each family mutually exclusive? — Use one label only; the strongest qualifying label wins

| Family | Exclusive? | Tie/overlap rule |
|---|---|---|
| Production style | `No` | `Players can offer combinations of production styles based on their positions (see production style table below)` |
| Usage state | Yes (`RECOMMENDED`) | Use one label only; the strongest qualifying label wins |
| Fixture behaviour | Yes (`RECOMMENDED`) | Use one label only; the strongest qualifying label wins|
| Venue behaviour | Yes (`RECOMMENDED`) | Use one label only; the strongest qualifying label wins |
| Return shape | Use one label only; the strongest qualifying label wins| Use one label only; the strongest qualifying label wins |
| Value state | Yes (`RECOMMENDED`) | Use one label only; the strongest qualifying label wins|
| Badges | No (`RECOMMENDED`) | Use one label only; the strongest qualifying label wins|



`Production style` Naming system

`G: Goal Threat` `C: Creator/Creativity`     `D: Defensive Activity/ Engine`    `S: Shot Stopping`    `W: Sweeping`    `P: Distribution`


| position | production style(s) | Display Name |
|---|---|---|
| `FWD` | `G` | `Finisher` |
| `FWD` | `G`, `C` | `Creative Forward` |
| `FWD` | `D` | `Pressing Forward` |
| `FWD` | `G`, `C` | `Complete Forward` |
| `FWD` | `G`, `D` | `Pressing Finisher` |
| `FWD` | `C`, `D` | `Pressing Link Forward` |
| `FWD` | `G`, `C`, `D` | `Complete Two-Way Forward` |
---
| position | production style(s) | Display Name |
|---|---|---|
| `MID` | `G` | `Goal-Scoring Midfielder` |
| `MID` |  `C` | `Playmaker` |
| `MID` |  `D` | `Ball-Winning Midfielder` |
| `MID` | `G`, `C` | `Attacking Playmaker` |
| `MID` | `G`, `D` | `Box-to-Box Midfielder` |
| `MID` | `C`, `D` | `Deep-Lying Playmaker` |
| `MID` | `G` | `Complete Midfielder` |
---
| position | production style(s) | Display Name |
|---|---|---|
| `DEF` | `G` | `Goal-Threat Defender` |
| `DEF` | `C` | `Creative Defender` |
| `DEF` | `D` | `Defensive Stopper` |
| `DEF` | `G`, `C` | `Attacking Defender` |
| `DEF` | `G`, `D` | `Two-Way Defender` |
| `DEF` | `C`, `D` | `Defensive Creator` |
| `DEF` | `G`, `C`, `D` | `Complete Defender` |
---
| position | production style(s) | Display Name |
|---|---|---|
| `GKP` | `S` | `Shot Stopper` |
| `GKP` | `W` | `Sweeper Keeper` |
| `GKP` | `P` | `Distributor` |
| `GKP` | `S`, `W` | `Proactive Shot Stopper` |
| `GKP` | `S`, `P` | `Ball-Playing Shot Stopper` |
| `GKP` | `W`, `P` | `Ball-Playing Sweeper` |
| `GKP` | `S`, `W`, `P` | `Complete Goalkeeper` |


---

## 3. Global archetype specification schema

Copy this block once for every `KEEP V1` scored archetype or state.

```text
ID:
Display name:
Family:
Definition in one sentence:
V1 status:
Eligible FPL positions:
Eligible player population:
Required raw fields and provider definitions:
Metric directions (higher/lower is better):
Rate denominators (per 90/per appearance/per start/etc.):
Context adjustments:
Metric preprocessing/caps:
Component standardization:
Component weights or fitted method:
Raw score equation:
Final score scale:
Reference population:
Historical window:
Earlier-current-season window:
Recent window:
Temporal combination formula:
Minimum total minutes:
Minimum eligible appearances:
Minimum starts (if relevant):
Minimum context samples (if relevant):
Small-sample shrinkage:
Label qualification threshold:
Exit threshold/hysteresis:
Confidence formula and display bands:
Suppression rule at low confidence:
Trend formula and thresholds:
Allowed statuses:
Within-family exclusivity/precedence:
Transfer/new-club handling:
Position-change handling:
Injury/absence handling:
Missing-data fallback:
Output fields:
Unit tests:
Backtest target and acceptance criterion:
Owner/version/effective date:
```

---

## 4. Worked sample specification — Goal Threat

This is a **fully worked illustrative example**, not a silently agreed definition. Replace every value if it does not match the intended design.

### 4.1 Identity and eligibility

- **ID:** `goal_threat`
- **Family:** Production style
- **Definition:** Sustained, position-relative tendency to generate high-quality shooting opportunities and goals while on the pitch.
- **Eligible positions:** DEF, MID, FWD. GK excluded.
- **Output:** continuous score `0–100`, confidence `0–1`, trend delta, status, and boolean display label.

### 4.2 Inputs and score

Use non-penalty process metrics as the core and realized goals as a smaller outcome component:

| Component | Transform | Illustrative weight |
|---|---|---:|
| non-penalty xG per 90 | Winsorized, then position-season z-score | 0.45 |
| shots in box per 90 | Winsorized, then position-season z-score | 0.25 |
| shots on target per 90 | Winsorized, then position-season z-score | 0.20 |
| non-penalty goals per 90 | Winsorized, then position-season z-score | 0.10 |

Illustrative window score:

$$
GT_z=0.45z(npxG90)+0.25z(SiB90)+0.20z(SOT90)+0.10z(npG90)
$$

Convert `GT_z` to a percentile against eligible players in the same FPL position and scoring date's reference population:

$$
GoalThreatScore=100\times \widehat F_{position,t}(GT_z)
$$

Rules:

- Penalties are excluded from the core score so penalty duty does not masquerade as open-play role. A separate penalty-taker flag may be displayed.
- A five-minute cameo contributes its underlying events and minutes but does not count as one full appearance in evidence checks.
- Components are standardized using training/past data only during backtests.
- If a required metric is unavailable, do not silently reweight unless the missing-data policy explicitly allows it.

### 4.3 Time model

Calculate three component scores using the same metric definition:

- `P`: previous-season score;
- `C`: earlier-current-season score, excluding the latest 10 eligible appearances;
- `R`: latest-10-eligible-appearances score.

Illustrative evidence-aware combination:

$$
w_R=0.50\frac{m_R}{m_R+450}
$$

$$
w_C=0.35\frac{m_C}{m_C+900}
$$

$$
w_P=1-w_R-w_C
$$

and

$$
GT_{current}=w_P P+w_C C+w_R R
$$

If no previous-season evidence exists, replace `P` with the position prior and cap confidence. These constants are examples; the filled workbook must either approve them or replace them.

### 4.4 Evidence, label, confidence, and trend

Illustrative rules:

- Full-label eligibility: at least `900` weighted minutes and `10` eligible appearances across retained evidence.
- Provisional eligibility: `450–899` weighted minutes and at least `6` eligible appearances.
- Insufficient evidence: below either provisional threshold.
- Label enters at `GoalThreatScore >= 80`.
- Label exits only below `75` to reduce threshold flicker.

Illustrative confidence:

$$
C_m=\min(1,minutes/1350)
$$

$$
C_a=\min(1,eligible\ appearances/15)
$$

$$
C_s=\max(0,1-|R-P|/50)
$$

$$
Confidence=0.55C_m+0.25C_a+0.20C_s
$$

The stability term must not punish a genuine role change indefinitely; cap its influence as above and expose the change through trend.

Illustrative trend:

$$
TrendDelta=R-H
$$

where `H` is the minute-weighted blend of previous and earlier-current evidence.

| Status | Illustrative rule |
|---|---|
| Emerging | historical score `<80`, current score `>=80`, trend `>=+10`, confidence `>=0.50` |
| Established | historical and current score `>=80`, confidence `>=0.70` |
| Stable | absolute trend `<5` and confidence `>=0.50` |
| Declining | current score is at least `10` below historical score |
| Provisional | score qualifies but confidence is `0.35–0.69` |
| Insufficient Evidence | confidence `<0.35` or evidence floor fails |

### 4.5 Worked numeric player example

Suppose an eligible midfielder has these position-relative window scores:

| Window | Score | Minutes |
|---|---:|---:|
| Previous season `P` | 74 | 1,800 |
| Earlier current `C` | 78 | 720 |
| Recent 10 `R` | 91 | 810 |

Using the illustrative formulas:

$$
w_R=0.50\times\frac{810}{1260}=0.321
$$

$$
w_C=0.35\times\frac{720}{1620}=0.156
$$

$$
w_P=1-0.321-0.156=0.523
$$

$$
GT_{current}=0.523(74)+0.156(78)+0.321(91)=80.1
$$

The player enters the Goal Threat label at `80.1`. Because the recent score exceeds the older baseline by more than 10 points, the likely status is **Emerging Goal Threat**, subject to the final confidence calculation.

### 4.6 Goal Threat decisions to return

- [ ✓ ] Accept/replace inputs.
- [ ✓ ] Accept/replace metric definitions and penalty treatment.
- [ ✓ ] Accept/replace component weights or choose a learned weighting method.
- [ ✓ ] Accept/replace position-relative reference population.
- [ ✓ ] Accept/replace temporal formula and constants.
- [ ✓ ] Accept/replace minute/appearance evidence floors.
- [ ✓ ] Accept/replace label entry and exit thresholds.
- [ ✓ ] Accept/replace confidence formula and cutoffs.
- [ ✓ ] Accept/replace trend/status rules.
- [ ✓ ] Specify tests and out-of-sample success criterion.

---

## 5. Player temporal model

### 5.1 Observation definitions

- Eligible appearance minimum minutes: 30 minutes
- Does a start below that floor count? No; it must meet the minute floor
- Are event rates aggregated by total events/total minutes (`RECOMMENDED`) or averaged match rates?Total events divided by total minutes
- Recent window: latest `10` eligible appearances Latest 10 eligible appearances
- Maximum calendar age of an observation: One season
- Treatment of stoppage-time/provider rounding: Use the provider's recorded minutes as supplied
- Treatment of extra time/cup matches: Exclude cup matches and extra time
- Competitions included: Premier League only

### 5.2 Historical retention and weights

**Selected V1 mechanism:** Evidence-dependent shrinkage; weights rise with minutes and appearances.

For each family:

| Family | Update speed | Previous-season weight/rule | Earlier-current rule | Recent rule or half-life | Freeze date? |
|---|---|---|---|---|---|
| Production | Medium | `30` | `30` | `40` | No fixed freeze; weights change with evidence|
| Usage | Very fast | `20` | `35` | `45` | No fixed freeze; weights change with evidence|
| Fixture | Very slow | `10` | `40` | `50` | No fixed freeze; weights change with evidence|
| Venue | Slow | `15` | `40` | `45` | No fixed freeze; weights change with evidence|
| Return shape | Medium/slow | `20` | `30` | `50` | No fixed freeze; weights change with evidence|
| Value | Medium | `20` | `40` | `40` | No fixed freeze; weights change with evidence|
| Risk badge | `very fast` | `20` | `40` | `40` | No fixed freeze; weights change with evidence|

When does current-season evidence fully replace the previous-season prior? After 15 eligible appearances

Is recency based on appearances, team matches, gameweeks, days, or minutes? Eligible player appearances

Missing gameweeks due to injury: decay the player's profile, hold it fixed, or increase uncertainty only? Keep the score but lower confidence

---

## 6. Position-relative scoring

- `AGREED` Compare players within FPL position by default.
- Position source and effective timestamp: Official FPL position at the scoring date
- Reference population: all registered players, players above an evidence floor, or active squad players? All registered players who meet the evidence floor
- Standardization unit: season, rolling date, multi-season, or model-training fold? Recalculate at each scoring date using past data
- Method: percentile rank, empirical CDF, robust z-score, normal z-score, or learned calibration? Percentile for display and z-score for calculations
- How ties are ranked: Break ties using more minutes
- Whether position thresholds differ beyond relative scoring: Yes, where football meaning differs by position
- Whether goalkeepers have a separate family/output schema: Yes; use a goalkeeper-specific schema
- Whether Defensive Engine applies across DEF/MID/FWD and how provider position/role affects it: DEF, MID, and FWD with position-relative scoring
- Minimum number of eligible peers required to calculate a percentile: 30 players
- Fallback if a position group is small: Use a rolling two-season position group

Leakage rule: scaling parameters and peer distributions for historical predictions must use only information available at the prediction cutoff.

---

## 7. Evidence and minutes thresholds

### 7.1 Global defaults

| Rule | Selected value |
|---|---|
| Minimum minutes for any score | `25% of total available minutes` |
| Minimum minutes for provisional label | 450 minutes|
| Minimum minutes for full label | 900 minutes |
| Minimum eligible appearances | 10 appearances |
| Minimum starts for start-based states | 6 starts |
| Maximum single-match weight | No more than 15% of the score |
| Minimum recent-window minutes | `25% of total available minutes` |
| Shrinkage target | `position mean` |
| Shrinkage strength | Treat 900 prior minutes as the default |

### 7.2 Archetype-specific evidence

| Family/archetype | Total minutes | Appearances | Relevant-context count | Both sides of comparison? | Other requirement |
|---|---:|---:|---:|---|---|
| Goal Threat | 900 minutes for a full label| 10 eligible appearances| n/a | n/a | No extra requirement|
| Creator | 900 minutes for a full label| 10 eligible appearances| n/a | n/a | No extra requirement|
| Defensive Engine | 900 minutes for a full label| 10 eligible appearances| n/a | n/a | No extra requirement|
| Clean-Sheet Dependent | 900 minutes for a full label| 10 eligible appearances| At least 10 matches with 60+ minutes| Yes; require evidence on both sides| No extra requirement|
| Save Machine | 900 minutes for a full label| 10 eligible appearances| shots faced: At least 30 shots faced| n/a | No extra requirement|
| Usage states | 900 minutes for a full label| 10 eligible appearances| team matches available: At least 8 team matches while available| n/a | No extra requirement|
| Fixture behaviour | 900 minutes for a full label| 10 eligible appearances| easy: At least 6 easy and 6 hard fixtures; hard: At least 6 easy and 6 hard fixtures| Yes (`RECOMMENDED`) | No extra requirement|
| Venue behaviour | 900 minutes for a full label| 10 eligible appearances| home: At least 8 home and 8 away matches; away: At least 8 home and 8 away matches| Yes | No extra requirement|
| Return shape | 900 minutes for a full label| 10 eligible appearances| returns/hauls: At least 15 appearances and 2 returns/hauls| n/a | No extra requirement|
| Value states | 900 minutes for a full label| 10 eligible appearances| price observations: Use start-of-gameweek price only | n/a | No extra requirement|
| Points Hazard | 900 minutes for a full label| 10 eligible appearances| cards/fouls: At least 900 minutes or 10 card/foul events | n/a | No extra requirement|

**Selected low-evidence output:** Show the score plus `Insufficient Evidence`, but do not show the label.

---

## 8. Confidence model

Confidence must describe evidence strength/uncertainty, not player quality.

### 8.1 Inputs

For each item mark `USE`, `DO NOT USE`, and weight/formula:

| Confidence ingredient | Decision | Formula/weight |
|---|---|---|
| Total relevant minutes | Use it in confidence | Use with a 15% weight |
| Eligible appearances | Use it in confidence | Use with a 15% weight |
| Starts/team opportunities | Use it in confidence | Use with a 15% weight |
| Relevant context counts | Use it in confidence | Use with a 15% weight |
| Balance across contexts | Use it in confidence | Use with a 15% weight |
| Standard error/posterior uncertainty | Use it in confidence | Use with a 15% weight |
| Historical/recent agreement | Use it in confidence | Use with a 15% weight |
| Data completeness/provider quality | Use it in confidence | Use with a 15% weight |
| Time since last appearance | Use it in confidence | Use with a 15% weight |
| Club/manager/role continuity | Use it in confidence | Use with a 15% weight |

### 8.2 Display and suppression

| Confidence band | Numeric range | UI behavior |
|---|---:|---|
| Insufficient | Use the evidence floor instead of a fixed range | Hide the whole archetype |
| Low / Provisional | 0.30–0.49| Hide until Medium confidence |
| Medium | 0.60–0.79 | Show label, score, and Medium confidence |
| High | 0.80–1.00 | Show full explanation |

- Hard suppression threshold: Hide the label below 0.35 confidence
- Maximum confidence for imported/no-PL-history players: Cap at 0.50 until 450 PL minutes
- Maximum confidence immediately after transfer/position change: Lower only context-sensitive labels
- Whether confidence is calibrated against empirical label stability: Yes, against future label stability

---

## 9. Trend and status

Candidate trend definition:

$$
Trend=RecentScore-HistoricalScore
$$

- Exact definition of `HistoricalScore`: Blend previous season and older current-season evidence
- Minimum recent evidence before a trend is shown: 450 minutes and 6 appearances
- Whether trend uses absolute score points, standard deviations, or credible probability: Difference in standard deviations
- Noise/dead band: ±5 score points
- Rising threshold: At least +0.5 standard deviations
- Declining threshold:  At most −0.5 standard deviations
- Number of updates a status must persist before display: 2 updates
- Entry/exit hysteresis for every displayed label: Use a 5-point gap around each label's threshold

| Status | Final mathematical rule |
|---|---|
| Established | Historical and current scores qualify; confidence at least 0.70 |
| Emerging | Current qualifies, historical did not, and trend is at least +10 |
| Stable | Trend stays within ±3 points |
| Declining | Recent score is at least 10 points lower |
| Provisional | Score qualifies but confidence is 0.35–0.69 |
| Insufficient Evidence | Confidence below 0.35 or evidence floor fails |

Can a label be both `Established` and `Declining`, or must status be exclusive? Use one label only; the strongest qualifying label wins

---

## 10. Usage states

Choose observed-history inputs, predictive-minutes-model inputs, or both:

- start rate: `USE`
- expected minutes: `USE`
- minutes per team match available: `USE`
- completion rate conditional on starting: `USE`
- cameo probability/rate: `USE`
- squad availability/injury flags: `USE`

| State | `p_start` range | Expected/observed minutes rule | Completion/cameo rule | Precedence |
|---|---:|---|---|---:|
| Nailed | At least 0.90 | At least 75 expected minutes | Completes 80% of starts; cameo risk below 10% | Resolve by expected minutes |
| Regular Starter | 0.70–0.89 | 55–69 | No extra completion rule | Resolve by expected minutes |
| Rotation Risk | p_start 0.35–0.64 | 30–59 expected minutes | Use high start-probability uncertainty | Resolve by expected minutes |
| Impact Sub | p_start below 0.35 with cameo probability at least 0.35 | Use cameo probability instead of expected minutes | Appears from the bench in at least half of available matches | Resolve by expected minutes |
| Fringe | Fewer than 3 appearances in the latest 10 team matches | Fewer than 15 expected minutes | Fewer than 3 cameos in 10 team matches | Resolve by expected minutes |
| 90-Minute Man (if kept) | Remove this label from V1 | At least 80 expected minutes | Remove this label from V1 | Remove this label from V1 |

- Lookback and decay: Latest 8 team matches with a 4-match half-life
- How DNPs while injured/suspended differ from selection DNPs: Exclude injury/suspension DNPs; count selection DNPs
- How team matches before player registration count: Do not count them
- Response to one unexpected benching/start: Lower usage score slightly; do not change label immediately
- Manager-change reset/uncertainty rule: Keep score but lower confidence for 5 matches

---

## 11. Fixture sensitivity

`AGREED` Replace crude FDR buckets with a continuous team/opponent matchup where possible.

Candidate model:

$$
Performance_{p,t}=\alpha_p+\beta_p Matchup_t+\gamma^TX_{p,t}+\epsilon_{p,t}
$$

where `X` controls for minutes/role, venue, team strength, and other agreed context. Player slopes should be shrunk toward position/role averages.

### 11.1 Decisions

- Matchup definition for attackers: Team attack minus opponent defence plus venue
- Matchup definition for defenders/GKs: Team defence minus opponent attack plus venue
- Response variable: FPL points/90, xGI/90, return probability, production score, or separate by role: Use xGI/90 for attackers and clean-sheet/save outcomes for DEF/GK
- Minutes treatment/offset: Use minutes as exposure and require 30 minutes
- Controls: Control for team strength and venue only
- Linear slope, difficulty bands, spline, or hierarchical model: Hierarchical linear slope
- Shrinkage method/strength: Shrink player slopes toward the position average Choose shrinkage by walk-forward testing
- Minimum range of opponent strength observed: Player must face opponents across at least 60% of the league-strength range
- Minimum easy, medium, and hard samples: 6 matches in each band
- Easy/hard definitions if bands remain: Bottom/top 25% of matchup scores for UI and evidence checks Use continuous scores and no bands for modelling
- Fodder Hunter threshold: Performance drops by at least 0.5 SD from easy to hard fixtures
- Fixture Proof equivalence band: Hard-versus-easy difference within ±0.20 SD
- Big-Game threshold if promoted from deferred: Keep deferred in V1
- Confidence and statistical/practical significance rule: Require confidence at least 0.70 and a meaningful effect
- Stability requirement across seasons/folds: Same direction in at least 2 seasons/folds **+** Positive out-of-sample improvement overall

Important: “not statistically significant” is not evidence of Fixture Proof. Define a practically small equivalence interval and require adequate evidence across the difficulty range.

---

## 12. Venue behaviour

Candidate adjusted effect:

$$
VenueEffect_p=AdjustedPerformance_{home}-AdjustedPerformance_{away}
$$

- Performance measure: Adjusted FPL points/90
- Adjustment for opponent strength, team strength, minutes, and fixture congestion: 900 minutes for a full label
- Home and away minimum minutes/appearances: 900 minutes for a full label
- Shrinkage toward position/league home effect: Shrink toward the position-average home effect **+** Choose shrinkage by stability testing
- Home Comfort threshold: Home effect at least +0.35 SD
- Road Warrior threshold: Away effect at least +0.35 SD
- Venue Neutral/Proof equivalence band: Home-away difference within ±0.20 SD
- Neutral-venue matches: exclude / separate / map — Exclude them
- Club transfer interaction (old and new home environments): Keep 50% of old venue evidence and lower confidence
- Whether venue labels are mutually exclusive: Use one label only; the strongest qualifying label wins

Naming decision: reserve **Road Warrior** for superior away performance. Use **Venue Proof/Neutral** for no meaningful venue effect. Do not use “Take Him Anywhere” for both.

---

## 13. Return shape

Define first:

- `Return`: `At least 5 FPL points
- `Blank`: 2 or fewer FPL points after at least 60 minutes.
- `Haul`: `10 or more FPL points 
- `Eligible appearance`: A start or at least 30 minutes
- Whether captaincy points are excluded: Exclude captaincy multipliers

### 13.1 Explosive

- Score: `P(Haul | eligible appearance)` / alternative Position-relative percentile of the player’s small-sample-adjusted probability of scoring at least ten FPL points in an eligible appearance.
- Position-relative normalization: Percentile within FPL position but retain adjusted score internally
- Label threshold: Explosive score at least 80, adjusted haul probability at least 10%, and confidence at least 0.50.
- Minimum haul opportunities/events: provisional at 15 eligible appearances and 2 hauls; full at 25 appearances and 3 hauls.

### 13.2 Reliable

Do not define reliability as low variance alone—a consistently poor player is not a useful Reliable archetype.

- Required mean/median production floor: At least the 60th positional percentile
- Return-rate/floor metric: Share of appearances with at least 5 points
- Dispersion/downside metric: Blank rate only
- Combined score/threshold: At least 5 FPL points

### 13.3 Deferred return ideas

- **Streaky:** specify lagged conditional-return effect after controlling for fixture/minutes before promotion to V1.
- **Boom-or-Bust:** specify minimum haul rate **and** blank rate, plus overlap rule with Explosive.

Out-of-sample persistence test required to promote either item: Keep both ideas deferred through V1

---

## 14. Value logic

`AGREED` Value is position-relative and must consider production alongside price. Price alone is insufficient.

Choose the base outcome:

- **Selected base value outcome:** Historical points per million.

### 14.1 Price and production decisions

- Price timestamp: purchase price/current price/start-of-GW/average: Price at the start of each gameweek
- Production window: evidence-weighted realized history for delivered value; expected points over the next 5 gameweeks for forward value.
- Expected or realized production: Display both as separate value views
- Position-specific replacement price/player: Best regularly playing player near the position's base price
- Minimum minutes/selection viability: At least 450 minutes and 50 expected minutes
- Bench/playing-time adjustment: show per-90 and expected-minutes-adjusted value separately; apply expected minutes only once.
- Treatment of price changes caused by transfers: Use start-of-gameweek price with no retroactive changes
- Free budget/base-price treatment: use value over a position-specific replacement player.

### 14.2 2×2 state thresholds

| State | Price band | Production/value band | Final rule |
|---|---|---|---|
| Sleeper Agent | Cheap | High | Cheap and value score at or above 80th percentile |
| Budget Fodder | Cheap | Low | Bottom 25% price and below-median production |
| Worth the Hype | Expensive | High | Top 25% price and top 20% production |
| High Maintenance | Expensive | Low/mediocre | Top 25% price and below-median value |

- Cheap threshold: Bottom 25% of position prices
- Expensive threshold: Top 25% of position prices
- High production/value threshold: At least 80th positional percentile
- Middle-band behavior (no label or nearest state): Show no value label
- Entry/exit hysteresis: Enter at 80 and exit below 75

---

## 15. Discipline/risk badge

Candidate concept:

$$
DisciplineRisk=P(Yellow)+w_RP(Red)+w_SP(SuspensionProximity)+\cdots
$$

- Inputs: yellow cards, straight reds, second yellows, fouls, suspension proximity, own goals, penalties conceded, missed penalties: Yellow, second-yellow, straight-red, and suspension proximity
- Weights/cost basis: expected FPL point loss or fitted weights: Expected FPL point loss
- Per-90 versus per-appearance denominator: Per 90 with minutes shrinkage
- Position/referee/context adjustment: Adjust for position only
- Evidence floor and shrinkage: 900 minutes, shrunk to position average
- Points Hazard threshold: Top 15% discipline-risk score by position
- Whether active suspension is a separate availability flag: Yes, separate availability flag

---

## 16. New, promoted, and transferred players

### 16.1 Player-history source hierarchy

Rank accepted evidence sources:

1. Previous Premier League history
2. Translated major-league/lower-league history
3. Position-and-role prior

Selected evidence-source rules:

- previous Premier League seasons: Use with recency decay
- promoted-club Championship history: Use with a league-strength translation
- other domestic leagues: Use only with a validated league translation
- European/cup competitions: Use only when league evidence is sparse
- youth/reserve football: Do not use in V1
- position/role prior only: Use when no trusted senior data exists

### 16.2 Translation and priors

| Case | Starting score prior | Translation factor/model | Confidence cap | Faster update? |
|---|---|---|---:|---|
| Promoted player with lower-league data | Position-and-role average | Validated league-and-age translation | 0.50 until 600 PL minutes | Yes, double recent-evidence weight for 5 appearances |
| Imported player from another league | Position-and-role average | Validated league-and-age translation | 0.50 until 600 PL minutes | Yes, double recent-evidence weight for 5 appearances |
| Rookie/no senior history | Position-and-role average | n/a | 0.50 until 600 PL minutes | Yes, double recent-evidence weight for 5 appearances |
| Returning PL player | Position-and-role average | age/recency: Validated league-and-age translation | Validated league-and-age translation | Validated league-and-age translation |

Recommended fallback:

$$
NoHistory\rightarrow Position/RolePrior+LowConfidence
$$

Do not force a strong label from a prior alone.

### 16.3 Transfers and role changes

- Within-PL transfer: retain 80% of player-skill evidence and 50% of context-sensitive evidence.
- Cross-league transfer: Translate old evidence and cap confidence at 0.50
- Confidence penalty and recovery schedule: Reduce confidence by 25%; recover over 5 appearances
- New manager without transfer: Keep scores but lower usage/context confidence for 5 matches
- Material role/set-piece change detector: Trigger after 3 consecutive changed-role starts
- Club-specific labels (venue, fixture, clean-sheet dependence) reset/shrink rule: Keep 50% and lower confidence

---

## 17. FPL position changes

- Recalculate all historical percentiles under the new position, preserve old scores, or blend: Keep raw events and recalculate against the new position
- Effective date: new FPL season / official reclassification date / Official FPL reclassification date
- Which raw per-90 evidence transfers unchanged: Transfer all raw events and rates
- Which labels reset because their semantics depend on position: Reset value, clean-sheet, and position-relative labels
- Confidence penalty: Reduce by 20% until 5 new-position appearances
- Threshold recovery evidence: 5 appearances or 450 minutes
- Hybrid-role/misclassification handling: Use official FPL position and store provider role separately
- Preserve an audit trail of old-position scores? Yes, preserve old and new scores

Recommended principle: preserve raw events, but re-standardize them against the new positional reference group; reduce confidence until enough new-position evidence accumulates.

---

## 18. Injury, absence, data, and other edge cases

For each case state score, confidence, label, and recovery behavior.

| Case | Score behavior | Confidence behavior | Label/UI behavior | Recovery rule |
|---|---|---|---|---|
| Long injury return | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Short injury | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Suspension | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| DNP while available | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Five-minute cameo | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Red-card match | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Team played with red card | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Abandoned/postponed match | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Double gameweek | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Missing provider metric | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Provider definition changes | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Extreme outlier | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Mid-season league registration | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |
| Club/manager tactical shock | Keep the last score; update only from valid new evidence | Lower confidence until enough new evidence arrives | Show score and confidence band | Return to normal after 5 valid appearances |

Selected additional rules:

- Outlier cap/winsorization level: Cap at the 1st and 99th percentiles
- Red-card minute/context adjustment: Exclude post-red-card team minutes and keep pre-card evidence
- Penalty-event treatment by archetype: Exclude penalties from open-play archetypes; show duty separately
- Data-quality failure policy: fail closed / last known score / partial score — Use the last valid score and raise a warning
- Backfill/revision policy when provider data changes: Recalculate affected snapshots and increase the data version

---

## 19. Team attack/defence strength V1

This section operationalizes the existing `FPL_player_archetype_system.md` team-context extension.

### 19.1 Meaning, sign, and scale

- Higher attack rating means: stronger attack (confirmed)
- Higher defence rating means: stronger defensive resistance (confirmed)
- Latent scale: Mean 0, where positive is stronger
- Display scale: index centred on 1.00.
- Overall Elo role: hierarchical prior (`RECOMMENDED`) / residual baseline / direct learned predictor — Hierarchical prior/anchor
- Constraint/consistency relationship between overall, attack, and defence: Shrink attack and defence jointly toward overall Elo
- Home-advantage representation and initial value: One learned league-wide home term

### 19.2 V1 attack metric families

Select a small interpretable baseline before adding correlated metrics.

| Family | Candidate metric(s) | Keep? | Provider/definition | Rate/context | Within-family weight |
|---|---|---|---|---|---:|
| Chance quality | non-penalty xG; open-play xG | Keep in V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Shot volume | shots in box; SOT | Keep in V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Chance creation | xA; key passes; big chances created | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Territory | box entries/touches; deep completions | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Possession value | xT/OBV-type metric | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Transition threat | transition xG/shots | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Set-piece attack | set-piece xG/shots | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Finishing outcome | goals/conversion/goals-xG | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |

Recommended minimum baseline: non-penalty xG plus shots in box, grouped so correlated metrics do not each receive a full independent vote.

### 19.3 V1 defence metric families

| Family | Candidate metric(s) | Keep? | Provider/definition | Orientation/context | Within-family weight |
|---|---|---|---|---|---:|
| Chance prevention | npxG conceded/open-play xGA | Keep in V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Shot prevention | shots/SIB/SOT conceded | Keep in V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Chance quality prevention | xG/shot; big chances conceded | Keep in V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Territory prevention | opp. box entries/touches | Defer from V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Possession-value prevention | xT/OBV conceded | Defer from V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Pressing/disruption | high turnovers; PPDA | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Transition defence | transition xG conceded | Defer from V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Set-piece defence | set-piece xG conceded | Defer from V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |
| Goalkeeping outcome | PSxG-GA/save performance | Defer from V1 | Use the project's current trusted provider and freeze its definition | Per match, opponent- and venue-adjusted | Equal weights inside the family |
| Final outcome | goals conceded/clean sheets | Defer from V1 | Use the project's current trusted provider and freeze its definition | reverse | Equal weights inside the family |

Whether goalkeeper shot-stopping belongs in team defensive strength, a separate keeper layer, or only the final forecast: Separate goalkeeper layer, then include it in the final forecast

### 19.4 Preprocessing and weighting

- Per match/per 90/per possession choice: Per match for team output
- Opponent adjustment: Adjust using pre-match opponent attack/defence ratings
- Home/away adjustment: Learn one league-wide home effect
- Red-card/numerical-state adjustment: Remove/down-weight minutes after a red card
- Game-state adjustment: Adjust for minutes leading/drawing/trailing
- Penalty treatment: Use non-penalty process metrics
- Rest/congestion adjustment: Add days-rest and short-turnaround flags
- Robust scaling/winsorization: Robust z-score plus 1st/99th-percentile caps
- Standardization population and window: All 20 teams using only data available at that date
- Correlation redundancy cutoff: Review pairs with absolute correlation above 0.85
- Family construction method: representative metric / equal composite / PCA / fitted — Equal composite of a few non-duplicate metrics
- Across-family weighting: equal (`baseline`) / ridge / elastic net / constrained model / other — Equal family weights as the baseline
- Prototype scale factors `60` and `40`: retain / retune / remove — Remove and use a defined display scale

Do not use univariate correlation coefficients as automatic weights. Use correlations to diagnose redundancy; choose predictive weights by chronological out-of-sample performance.

### 19.5 Season-start initialization

For established teams:

$$
E_{i,0}=\bar E+\rho_E(E_{i,prev}-\bar E),\quad
A_{i,0}=\rho_A A_{i,prev},\quad
D_{i,0}=\rho_D D_{i,prev}
$$

- `rho_E`: 0.75
- `rho_A`: 0.70
- `rho_D`: 0.70
- Preseason Elo timestamp/provider: Rating immediately before the first league match
- Previous-season metric cutoff: End of the previous PL season
- Squad/manager continuity adjustment: Lower confidence, not the rating, when change is large

For promoted teams, the discussion accepted relegated-team performance as the V1 baseline idea, but the exact mean should be shrunk and carry lower confidence.

- Prior source: relegated-team mean / median / shrunk mean / lower-division translated model — Relegated-team mean shrunk toward league average
- Metric-by-metric or final-score prior: Metric-by-metric before standardization
- Shrinkage target for the promoted-team prior: Premier League average.
- Shrinkage factor: 50% relegated-team prior, 50% league average
- Different priors for promoted teams based on finishing/playoffs: No; same prior but different Elo and uncertainty
- Initial uncertainty/confidence: 50% of established-team confidence
- Faster early update multiplier: 2.0×
- Number of matches before normal update speed: 8 matches
- Complete-20-team standardization order: Insert promoted-team priors, then standardize all 20 together

### 19.6 Dynamic updates

**Selected update method:** Opponent-adjusted Elo-like residual update.

Core candidate:

$$
A_{i,t+1}=A_{i,t}+K_A r_{i,j,t}
$$

$$
D_{j,t+1}=D_{j,t}-K_D r_{i,j,t}
$$

- Observed attacking signal: xG / npxG / blended process-outcome — Non-penalty xG
- Expected signal model/link: Log-link model of team attack versus opponent defence
- `K_A`: Choose separately by walk-forward validation
- `K_D`: Choose separately by walk-forward validation
- EWMA half-life if used: `5 matches
- Early-season update multiplier/schedule: 1.5× for the first 5 matches
- Promoted-team update multiplier/schedule: 2.0× for 8 matches
- Extreme residual cap: Cap at ±2.5 standard deviations
- Update timing: after every match once the data passes quality checks.
- Data-availability cutoff/time zone: Latest verified data 24 hours before kickoff, UTC
- Store immutable pre-match snapshots: Yes; store every pre-match snapshot.

### 19.7 Future targets

- Attack process target and horizon: Next-5-match non-penalty xG/xGA
- Attack outcome target and horizon: Next-5-match goals/goals conceded
- Defence process target and horizon: Next-5-match non-penalty xG/xGA
- Defence outcome target and horizon: Next-5-match goals/goals conceded
- Process/outcome blend `alpha`, if any: 0.70 process and 0.30 outcome
- Separate targets for rating construction versus FPL forecasting: Yes: process for ratings, FPL outcomes for final forecasts

### 19.8 Fixture multipliers for players

- Attacking matchup formula: Team attack − opponent defence + home effect
- Defensive matchup formula: Team defence − opponent attack + home effect
- Map latent matchup to multiplier/probability: Exponential/logistic mapping learned from history
- Cap/floor fixture multipliers: 0.70 to 1.30
- Player baseline definition: Evidence-weighted player per-90 production
- Role/minutes multiplier: Expected minutes divided by 90
- Set-piece/penalty-share adjustment: Add current estimated share as a separate adjustment
- Clean-sheet probability model: Poisson probability of zero opponent goals
- Save-volume interaction for GKs: Opponent shot volume × goalkeeper save rate

---

## 20. Validation specification

### 20.1 Data split and reproducibility

- Seasons available: Use every complete season with consistent data
- Initial training seasons/window: At least 2 full seasons
- Walk-forward validation unit: Next gameweek
- Refit/update cadence: Update ratings after every match; refit weights monthly
- Test seasons held out: Latest complete season
- Timestamp/data-cutoff convention: Only data available before kickoff
- Preprocessing fitted inside each fold: Yes (confirmed)
- Pre-match ratings/features stored: Yes (confirmed)
- Random seeds/versioning policy: Fixed seeds plus data/code/model version

### 20.2 Team-strength baselines

The V1 must be compared with:

- league-average expectation;
- recent goals/goals conceded;
- recent xG/xGA;
- overall Elo alone;
- simple rolling non-penalty xG/xGA;
- current official FDR if used in the product.

Add/remove baselines: Keep all listed baselines

### 20.3 Metrics

| Target | Primary metric | Secondary metrics | Acceptance criterion |
|---|---|---|---|
| Future xG/xGA | MAE/RMSE | rank correlation | Beat the strongest baseline by at least 2% |
| Goals/counts | Poisson deviance | MAE/RMSE | Beat the strongest baseline by at least 2% |
| Goal/return/CS probability | log loss | Brier score, calibration | Beat the strongest baseline by at least 2% |
| Team-strength ranking | Spearman | stability | Beat the strongest baseline by at least 2% |
| Archetype label | Use the metric already named in the row | precision/recall/stability | Beat the strongest baseline by at least 2% |
| Confidence | calibration | coverage | Beat the strongest baseline by at least 2% |
| Trend/status | future directional change | persistence | Beat the strongest baseline by at least 2% |

### 20.4 Archetype-specific validity

For every kept archetype, answer:

1. **Construct validity:** Does the score match its football meaning? Test: Expert review plus expected metric relationships
2. **Reliability:** Is it stable when no real change occurs? Test: Test score and label stability across adjacent windows
3. **Responsiveness:** Does it react to genuine role changes at the intended speed? Test: Test known role, manager, and club changes
4. **Predictive validity:** Does it predict relevant future behaviour? Target/horizon: Predict the next 5 eligible appearances
5. **Incremental value:** Does it add information beyond minutes, price, position, team strength, and raw inputs? Test: Compare with the same model using raw inputs but no archetype
6. **Calibration:** Does confidence correspond to empirical correctness/stability? Test: Compare confidence bands with future label stability
7. **Fair comparison:** Are positions, sample sizes, and context handled correctly? Test: Audit by position, minutes, club strength, and player origin

### 20.5 Promotion and rejection rules

- Minimum improvement over strongest simple baseline: At least 2% over the strongest simple baseline
- Required folds/seasons showing improvement: At least 2 seasons and 70% of folds
- Maximum allowed calibration degradation: No more than 1%
- Maximum label churn: review over 4 gameweeks; no more than 10% of labels may change.
- Minimum out-of-sample label persistence: At least 5 eligible appearances
- Complexity rule: add a feature/family only if It improves validation and has a clear football meaning
- Failure action: simplify / recalibrate / defer / remove — Simplify first, then recalibrate, otherwise defer

---

## 21. Output contract and governance

### 21.1 Suggested player-level output

```json
{
  "player_id": "...",
  "as_of": "YYYY-MM-DDTHH:MM:SSZ",
  "fpl_position": "MID",
  "archetypes": [
    {
      "id": "goal_threat",
      "family": "production",
      "score": 80.1,
      "label_active": true,
      "confidence": 0.74,
      "confidence_band": "high",
      "trend_delta": 12.4,
      "status": "emerging",
      "evidence_minutes": 3330,
      "eligible_appearances": 39,
      "version": "1.0.0"
    }
  ]
}
```

- Final score range/precision: 0–100 with one decimal
- Null versus omitted fields: Use null for known-but-unavailable; omit fields that do not apply
- Stable IDs and final display names: Recent and historical scores differ by less than 5 points
- Snapshot frequency: Before every match deadline
- Explanation fields shown to users: Score, label, confidence, trend, and main reasons
- Audit fields retained internally: Inputs, weights, evidence, timestamps, and model version
- Versioning policy: Semantic versioning: major.minor.patch
- Recalculation/backfill policy: Backfill only data corrections; preserve old model-version outputs
- Responsible owner/reviewer: Project owner

---

## 22. Deferred ideas and explicit non-goals

### 22.1 Recommended deferred ideas

- Big-Game Performer until player difficulty slopes are stable out of sample.
- Streaky until temporal clustering survives fixture/minutes controls.
- Boom-or-Bust if it cannot be separated cleanly from Explosive and Reliable.
- Bonus Magnet until incremental meaning beyond other production sources is shown.
- Clean Record inverse discipline badge.
- True separate attack-Elo and defence-Elo systems beyond the interpretable V1 composite/dynamic update.
- Nonlinear position-specific fixture-response curves.
- Central/wide/set-piece team-strength decompositions.
- Complex lower-league/cross-league translation models if reliable data is not ready.
- Feeding archetype labels into the predictive model before demonstrating incremental out-of-sample value.

### 22.2 Explicit removals/non-archetypes

- **Top Points Provider / Top-two Points Provider:** ranking/statistic only, not an archetype.
- Raw FDR bucket alone is not sufficient proof of fixture behaviour.
- Goals alone do not define Goal Threat.
- Assists alone do not define Creator.
- Low variance alone does not define Reliable.
- High same-period correlation with overall Elo does not justify team metric selection or weights.

### 22.3 Deferred-item promotion template

```text
Item:
Why it is currently deferred:
New evidence/data now available:
Exact proposed definition:
Minimum sample:
Out-of-sample target:
Baseline:
Required improvement/stability:
Overlap with existing labels:
Decision date and owner:
```

---

## 23. Final implementation-readiness checklist

Before returning this workbook, confirm every box below.

### Scope and naming

- [✓ ] Every discussed item has a final `KEEP V1`, `CONSIDER V1`, `DEFER`, or `REMOVE` status.
- [ ✓] Every kept item has one stable ID and one final display name.
- [ ✓] Family exclusivity, overlaps, and precedence are explicit.
- [ ✓] V1 non-goals and deferred items are frozen.

### Every kept archetype/state/badge

- [ ✓] One-sentence meaning and eligible positions are defined.
- [ ✓] All raw inputs, providers, units, and directions are defined.
- [ ✓] Score equation/learning method and final scale are defined.
- [ ✓] Reference population and position-relative method are defined.
- [ ✓] Historical, earlier-current, and recent windows are defined.
- [ ✓] Temporal weights/decay and update speed are numerical.
- [ ✓] Minutes, appearances, and context evidence floors are numerical.
- [ ✓] Small-sample shrinkage and missing-data fallback are defined.
- [ ✓] Entry threshold, exit threshold, and hysteresis are defined.
- [ ✓] Confidence formula, bands, cap, and suppression rule are defined.
- [ ✓] Trend formula, dead band, and every status rule are defined.
- [ ✓] Transfer, position-change, injury, and role-change behavior is defined.
- [ ✓] Unit tests and out-of-sample acceptance criteria are defined.

### Cross-cutting player rules

- [✓] “Eligible appearance,” “return,” “blank,” and “haul” are numerical.
- [✓] Current-season evidence replacement/retention is settled.
- [✓] DNP, cameo, red-card, missing-data, and outlier rules are settled.
- [✓] New/promoted/imported/rookie player priors and confidence caps are settled.
- [✓] FPL position-change logic is settled.
- [✓] Usage-state thresholds are non-overlapping and exhaustive or have a declared `Unclassified` state.
- [✓] Fixture Proof uses an equivalence band, not lack of significance.
- [✓] Venue labels adjust for opponent strength and minutes.
- [✓] Value logic specifies price timestamp, replacement/opportunity cost, and production window.

### Team attack/defence V1

- [✓] Rating meaning, sign, latent/display scale, and Elo role are fixed.
- [✓] Attack and defence metric families and providers are selected.
- [✓] Correlated metrics are grouped or removed.
- [✓] Preprocessing and every context adjustment are defined.
- [✓] Family/metric weighting method is defined.
- [✓] The `60/40` prototype scales are explicitly kept, retuned, or removed.
- [✓] Established-team mean reversion parameters are set.
- [✓] Promoted-team prior, shrinkage, uncertainty, and faster-update schedule are set.
- [✓] All 20 teams are inserted before final standardization.
- [✓] Rolling/EWMA/residual update method and all constants are set.
- [✓] Immutable pre-match ratings and data cutoffs are required.
- [✓] Fixture-to-player multiplier mapping and caps are defined.

### Validation and delivery

- [✓] Walk-forward folds, horizons, refit cadence, and held-out seasons are fixed.
- [✓] Baselines, metrics, and numerical pass/fail criteria are fixed.
- [✓] Confidence, label stability, trend usefulness, and subgroup behavior are tested.
- [✓] Output schema, versioning, audit trail, and owner are specified.
- [✓] Every former choice field now contains one selected value.
- [✓] A second reviewer can implement the system without making a new analytical choice.

---

## 24. Final sign-off

- Specification owner: Okechi Ukandu
- Statistical reviewer: Okechi Ukandu
- Data-source reviewer: Okechi Ukandu
- V1 version: 1.0.0
- Effective date: 08/16/2026
- Approved for implementation: `YES / NO`
- Known limitations accepted for V1: Accept the deferred items listed in Section 22

