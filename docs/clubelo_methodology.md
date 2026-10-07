# ClubElo Methodology Reference

> Status: conceptual methodology reference, not an executable implementation
> contract or a freshly verified description of the external service. The local
> [fallback contract](CLUBELO_FALLBACK_DESIGN.md) defines its intentionally smaller
> supported scope; the [combined runbook](COMPLETE_DATA_PIPELINE.md) covers acquisition.

This note stores the ClubElo-style logic you want to keep around for future work in this repo. It is a reference summary of the rating system, not a claim that every rule below is already implemented in code.

## 1. Core Elo model

Each club has a single Elo value at every point in time. Higher Elo means stronger team strength.

The expected result for team 1 against team 2 is:

```text
E = 1 / (10^(-dr / 400) + 1)
```

Where:

- `E` = expected result for team 1
- `dr` = Elo difference between the two teams
- draws count as half-win / half-loss in expectation

## 2. Base points exchange

After a match, Elo is updated with:

```text
Delta_Elo_1X2 = (R - E) * k
```

Where:

- `R = 1` for a win
- `R = 0.5` for a draw
- `R = 0` for a loss
- `k = 20`

Interpretation:

- larger `k` makes ratings react faster but adds more volatility
- smaller `k` makes ratings more stable but slower to converge

## 3. Goal-difference weighting

Winning by a larger margin should move ratings more than a narrow win. ClubElo scales updates with the square root of the margin:

```text
Delta_Elo_margin = Delta_Elo_1goal * sqrt(margin)
```

Where the 1-goal baseline is normalized as:

```text
Delta_Elo_1goal = Delta_Elo_1X2 / sum(sqrt(margin) * p_margin / p_1X2)
```

Definitions:

- `margin` = goal margin for the result
- `p_margin` = probability of a specific winning or losing margin
- `p_1X2` = probability of the overall win or loss outcome
- the sum runs over all win margins or all loss margins, depending on result

This keeps the expected Elo exchange consistent with the base Elo equation while still rewarding bigger margins.

## 4. Home field advantage

Home teams tend to perform better, so the pre-match Elo difference is adjusted by a home field advantage term, `HFA`.

Reason for the adjustment:

- without `HFA`, home teams would systematically gain Elo on average
- that would break the zero-sum expectation of Elo updates across matches

ClubElo updates `HFA` separately by country, once per day:

```text
HFA += sum(Delta_Elo) * 0.075
```

Interpretation:

- if home teams gained too many points, `HFA` increases
- if away teams gained too many points, `HFA` decreases

## 5. Tilt

ClubElo adds a second metric called `Tilt` to capture offensiveness, meaning whether matches involving a team produce more or fewer total goals than expected.

Important distinction:

- `Elo` measures quality / strength
- `Tilt` measures whether game totals run hotter or colder than expected

Tilt starts at `1.0` and updates after each match with:

```text
New_tilt = 0.98 * Old_tilt + 0.02 * (Game_total_goals / Opposition_tilt / Exp_Game_total_goals)
```

Where:

- `Game_total_goals` = actual total goals in the match
- `Opposition_tilt` = opponent's current tilt
- `Exp_Game_total_goals` = expected total goals for a match at that Elo difference

This is used to improve exact-score or result-distribution prediction.

## 6. Match odds / result histogram

ClubElo match odds are based on a result histogram derived from the Elo difference between the two clubs.

Practical interpretation:

- Elo difference drives win/draw/loss expectation
- margin probabilities and expected total goals refine scoreline-level prediction
- Tilt helps improve exact result modeling

## 7. Two-leg matches

Two-leg ties are treated as one long match.

Rules:

- the aggregate score over both legs determines the total Elo exchange
- compared with a single match, total exchange is multiplied by `sqrt(2)`
- winning on away goals counts as winning by a margin of half a goal
- the Elo points exchanged after leg 1 are based on the change in outcome likelihood from before leg 1 to after leg 1

## 8. Working summary

If you later implement this system in code, the flow is:

1. Start from current Elo values.
2. Adjust pre-match Elo difference with home field advantage.
3. Convert Elo difference into expected result using the Elo equation.
4. Compute base Elo exchange from actual result vs expected result.
5. Scale the exchange by goal margin using the square-root margin rule.
6. Update both clubs' Elo values.
7. Update Tilt using actual vs expected total goals.
8. Update country-level HFA over time.

## 9. Constants captured here

```text
k = 20
Tilt decay = 0.98
Tilt update weight = 0.02
HFA daily adjustment factor = 0.075
Two-leg multiplier = sqrt(2)
```
