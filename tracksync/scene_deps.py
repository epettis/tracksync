"""Scene-mode dependency guard.

This module provides utilities for checking and requiring scene-mode
dependencies (torch, timm, lightglue, scipy) without importing them at
module load time.
"""

import importlib.util


class MissingSceneDependenciesError(ImportError):
    """Raised when scene-mode dependencies are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "Scene alignment mode requires additional dependencies. "
            "Install them with: pip install -e '.[scene]'"
        )


def require_scene_deps() -> None:
    """Check that scene-mode dependencies are available.

    Raises:
        MissingSceneDependenciesError: If torch is not installed.

    Note:
        This only checks for torch as a proxy for the full scene extra,
        since torch is the primary heavy dependency. The other dependencies
        (timm, lightglue, scipy) are also required but will raise their own
        ImportErrors if missing when actually imported.
    """
    if importlib.util.find_spec("torch") is None:
        raise MissingSceneDependenciesError()
