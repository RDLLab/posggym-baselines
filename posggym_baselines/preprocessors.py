"""Preprocessors for policies."""
from collections.abc import Callable
from typing import Any

from gymnasium import spaces


ObsPreprocessor = Callable[[Any], Any]


def identity_preprocessor(obs: Any) -> Any:
    """Return the observation unchanged."""
    return obs


def get_flatten_preprocessor(obs_space: spaces.Space) -> ObsPreprocessor:
    """Get the preprocessor function for flattening observations."""

    def flatten_preprocessor(obs: Any) -> Any:
        """Flatten the observation."""
        return spaces.flatten(obs_space, obs)

    return flatten_preprocessor
