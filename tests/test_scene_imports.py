"""Test that scene modules import cleanly without torch installed.

These tests verify that:
1. All five scene modules can be imported without torch being installed
2. The require_scene_deps() guard properly detects missing dependencies
3. The error message mentions the 'scene' pip extra

This ensures that the base tracksync package remains usable without the
heavy ML dependencies, and that scene mode fails gracefully with a clear
error message when dependencies are missing.
"""

import importlib.util
import sys
from unittest.mock import patch

import pytest


def test_scene_deps_imports():
    """Test that scene_deps module imports cleanly."""
    from tracksync.scene_deps import MissingSceneDependenciesError, require_scene_deps

    assert MissingSceneDependenciesError is not None
    assert require_scene_deps is not None


def test_embedding_module_imports():
    """Test that embedding module imports without torch."""
    import tracksync.embedding
    assert tracksync.embedding is not None


def test_dtw_module_imports():
    """Test that dtw module imports without torch."""
    import tracksync.dtw
    assert tracksync.dtw is not None


def test_masking_module_imports():
    """Test that masking module imports without torch."""
    import tracksync.masking
    assert tracksync.masking is not None


def test_fine_align_module_imports():
    """Test that fine_align module imports without torch."""
    import tracksync.fine_align
    assert tracksync.fine_align is not None


def test_scene_align_module_imports():
    """Test that scene_align module imports without torch."""
    import tracksync.scene_align
    assert tracksync.scene_align is not None


def test_require_scene_deps_detects_missing_torch():
    """Test that require_scene_deps() raises error when torch is unavailable."""
    from tracksync.scene_deps import MissingSceneDependenciesError, require_scene_deps

    # Check if torch is actually available
    torch_available = importlib.util.find_spec("torch") is not None

    if torch_available:
        # If torch IS installed, temporarily hide it to test the error path
        with patch("importlib.util.find_spec") as mock_find_spec:
            # Make find_spec return None for torch, but work normally for others
            def patched_find_spec(name):
                if name == "torch":
                    return None
                return importlib.util.find_spec(name)

            mock_find_spec.side_effect = patched_find_spec

            with pytest.raises(MissingSceneDependenciesError) as exc_info:
                require_scene_deps()

            error_msg = str(exc_info.value)
            assert "scene" in error_msg.lower()
            assert "pip install" in error_msg.lower()
            assert "'.[scene]'" in error_msg or "[scene]" in error_msg
    else:
        # If torch is NOT installed (expected in this environment), test directly
        with pytest.raises(MissingSceneDependenciesError) as exc_info:
            require_scene_deps()

        error_msg = str(exc_info.value)
        assert "scene" in error_msg.lower()
        assert "pip install" in error_msg.lower()
        assert "'.[scene]'" in error_msg or "[scene]" in error_msg


def test_missing_deps_error_message_format():
    """Test that MissingSceneDependenciesError has the correct message."""
    from tracksync.scene_deps import MissingSceneDependenciesError

    error = MissingSceneDependenciesError()
    error_msg = str(error)

    # Verify the error message mentions:
    # 1. That scene alignment mode requires dependencies
    # 2. The pip install command
    # 3. The 'scene' extra specifically
    assert "scene" in error_msg.lower() or "Scene" in error_msg
    assert "pip install" in error_msg
    assert "[scene]" in error_msg
