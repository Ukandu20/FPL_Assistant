# Player production profiles

For the combined methodology and model workflow, see
[Archetyping and modelling](ARCHETYPING_AND_MODELLING.md).

> Scope: the legacy season production-profile artifact. This is distinct from
> [versioned V1 archetypes](FPL_ARCHETYPE_V1.md) and the
> [signal-profile design](FPL_PLAYER_SIGNAL_PROFILE_DESIGN.md).
> The example season is historical; choose the season you intend to publish.

The Player Card uses a published league-season artifact rather than computing
profiles inside Streamlit or importing notebook state.

## Publish a season

```powershell
fpl-player-profiles --league "ENG-Premier League" --season 2025-2026
```

The equivalent module command is:

```powershell
python -m fpl_assistant.providers.fpl.pipelines.player_profiles `
  --league "ENG-Premier League" `
  --season 2025-2026
```

The artifact is written to:

```text
data/processed/fpl/{league}/{season}/analytics/player_profiles.csv
```

## Method

Each count metric is converted to a per-90 rate and pulled toward the
minutes-weighted mean for the player's FPL position:

```text
reliability = minutes / (minutes + prior_minutes)
adjusted = reliability × player_rate + (1 - reliability) × position_mean
```

Adjusted metrics are standardized within position and combined into bounded
0–100 scores, percentiles, and ordinal ranks. Reliability is not multiplied
into the final composite a second time.

Profile states are:

- **Established:** at least 900 minutes.
- **Provisional:** 450–899 minutes.
- **Insufficient data:** fewer than 450 minutes.
- **Data unavailable:** required provider tables did not match the player.

Outfield profiles use goal threat, creativity, and defensive activity.
Goalkeeper profiles use shot stopping, sweeping, and distribution. Archetype
describes style, while production tier describes the strength of the player's
best dimension.

The Streamlit app joins profiles by canonical `player_id`. It never substitutes
missing provider rows with zero production and never falls back to another
season's profile or forecast.
