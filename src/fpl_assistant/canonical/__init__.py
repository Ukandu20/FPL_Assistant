"""Provider-independent identity, match, fact, and feature architecture."""

from .facts import (
    DEFAULT_PLAYER_SOURCE_POLICY,
    DEFAULT_TEAM_SOURCE_POLICY,
    FactBuildResult,
    build_canonical_facts,
)
from .identity import RegistryBuildResult, build_identity_registry, stable_canonical_id
from .matches import MatchRegistryResult, build_match_registry
from .staging import stage_match_facts

__all__ = [
    "DEFAULT_PLAYER_SOURCE_POLICY",
    "DEFAULT_TEAM_SOURCE_POLICY",
    "FactBuildResult",
    "MatchRegistryResult",
    "RegistryBuildResult",
    "build_canonical_facts",
    "build_identity_registry",
    "build_match_registry",
    "stable_canonical_id",
    "stage_match_facts",
]
