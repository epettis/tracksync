"""Tests for the scene-mode CLI integration (`sync --mode scene`).

Covers argparse-level flag parsing, catalyst-mode rejection of scene-only
flags, scene-mode default filling, missing-dependency exit behavior, main()
dispatch, and one end-to-end invocation on tiny synthetic clips using the
gist embedder and a test-only synthetic matcher.

Task reference: docs/scene_alignment_tasks.md T11
Design reference: docs/scene_alignment_design.md §6, §11.3
"""

import argparse
from pathlib import Path

import numpy as np
import pytest

from tracksync import cli
from tracksync.cli import create_parser, main, validate_sync_args
from tracksync.csv_reader import read_csv
from tracksync.scene_deps import MissingSceneDependenciesError

from tests.test_scene_align_coarse import create_synthetic_moving_noise_video


class TestSceneFlagParsing:
    """Argparse-level tests for the new sync flags."""

    def test_mode_defaults_to_scene(self):
        parser = create_parser()
        args = parser.parse_args(['sync', 'a.mp4', 'b.mp4'])
        assert args.mode == 'scene'

    def test_mode_scene_parses(self):
        parser = create_parser()
        args = parser.parse_args(['sync', 'a.mp4', 'b.mp4', '--mode', 'scene'])
        assert args.mode == 'scene'

    def test_mode_rejects_unknown_value(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['sync', 'a.mp4', 'b.mp4', '--mode', 'bogus'])

    def test_scene_flags_default_to_none(self):
        """Scene-only flags use None sentinels so catalyst can reject them."""
        parser = create_parser()
        args = parser.parse_args(['sync', 'a.mp4', 'b.mp4'])
        assert args.sample_hz is None
        assert args.band_pct is None
        assert args.min_inliers is None
        assert args.fov_deg is None
        assert args.matcher is None
        assert args.embedder is None

    def test_scene_flags_parse_with_types(self):
        parser = create_parser()
        args = parser.parse_args([
            'sync', 'a.mp4', 'b.mp4', '--mode', 'scene',
            '--sample-hz', '5', '--band-pct', '0.2',
            '--min-inliers', '15', '--fov-deg', '120',
            '--matcher', 'superpoint-lightglue', '--embedder', 'gist',
        ])
        assert args.sample_hz == 5.0
        assert args.band_pct == 0.2
        assert args.min_inliers == 15
        assert args.fov_deg == 120.0
        assert args.matcher == 'superpoint-lightglue'
        assert args.embedder == 'gist'

    def test_sync_help_documents_new_flags(self, capsys):
        """`tracksync sync --help` must document the scene flags."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['sync', '--help'])
        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out
        for flag in ('--mode', '--sample-hz', '--band-pct', '--min-inliers',
                     '--fov-deg', '--matcher', '--embedder'):
            assert flag in help_text, f"{flag} missing from sync --help"


class TestValidateSyncArgs:
    """Tests for validate_sync_args rejection and default-filling logic."""

    def _parse(self, extra):
        parser = create_parser()
        return parser.parse_args(['sync', 'a.mp4', 'b.mp4'] + extra)

    @pytest.mark.parametrize('flag_args', [
        ['--sample-hz', '10'],
        ['--band-pct', '0.1'],
        ['--min-inliers', '30'],
        ['--fov-deg', '90'],
        ['--matcher', 'aliked-lightglue'],
        ['--embedder', 'dinov2-vitb14'],
    ])
    def test_catalyst_rejects_scene_only_flags(self, flag_args, capsys):
        """Every scene-only flag must be rejected in catalyst mode (exit 2)."""
        args = self._parse(['--mode', 'catalyst'] + flag_args)
        with pytest.raises(SystemExit) as exc_info:
            validate_sync_args(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert flag_args[0] in err
        assert '--mode scene' in err

    def test_catalyst_rejects_multiple_flags_listing_all(self, capsys):
        args = self._parse(['--mode', 'catalyst', '--sample-hz', '10', '--fov-deg', '90'])
        with pytest.raises(SystemExit) as exc_info:
            validate_sync_args(args)
        assert exc_info.value.code == 2
        err = capsys.readouterr().err
        assert '--sample-hz' in err
        assert '--fov-deg' in err

    def test_catalyst_without_scene_flags_is_noop(self):
        args = self._parse(['--mode', 'catalyst'])
        validate_sync_args(args)  # Must not raise
        assert args.mode == 'catalyst'
        assert args.sample_hz is None

    def test_scene_mode_fills_defaults(self):
        args = self._parse(['--mode', 'scene'])
        validate_sync_args(args)
        assert args.sample_hz == 10.0
        assert args.band_pct == 0.10
        assert args.min_inliers == 30
        assert args.fov_deg == 90.0
        assert args.matcher == 'aliked-lightglue'
        assert args.embedder == 'dinov2-vitb14'

    def test_scene_mode_preserves_explicit_values(self):
        args = self._parse([
            '--mode', 'scene', '--sample-hz', '4', '--embedder', 'gist',
        ])
        validate_sync_args(args)
        assert args.sample_hz == 4.0
        assert args.embedder == 'gist'
        # Untouched flags still get defaults
        assert args.min_inliers == 30
        assert args.matcher == 'aliked-lightglue'


class TestCheckSceneDeps:
    """Tests for the missing-dependency guard."""

    @staticmethod
    def _args(embedder, matcher):
        return argparse.Namespace(embedder=embedder, matcher=matcher)

    def test_torch_backends_exit_2_when_deps_missing(self, monkeypatch, capsys):
        def mock_require():
            raise MissingSceneDependenciesError()

        monkeypatch.setattr(cli, 'require_scene_deps', mock_require)

        for embedder, matcher in [
            ('dinov2-vitb14', 'synthetic'),
            ('gist', 'aliked-lightglue'),
            ('dinov2-vits14', 'superpoint-lightglue'),
        ]:
            with pytest.raises(SystemExit) as exc_info:
                cli.check_scene_deps(self._args(embedder, matcher))
            assert exc_info.value.code == 2
            err = capsys.readouterr().err
            assert "pip install -e '.[scene]'" in err

    def test_gist_and_synthetic_skip_dep_check(self, monkeypatch):
        def mock_require():
            raise AssertionError("require_scene_deps must not be called")

        monkeypatch.setattr(cli, 'require_scene_deps', mock_require)
        cli.check_scene_deps(self._args('gist', 'synthetic'))  # Must not raise

    def test_deps_checked_before_any_decoding(self, monkeypatch, capsys):
        """CLI must exit 2 before feature extraction or decoding starts."""
        def mock_require():
            raise MissingSceneDependenciesError()

        def fail_extract(*a, **k):
            raise AssertionError("decoding started before deps check")

        monkeypatch.setattr(cli, 'require_scene_deps', mock_require)
        monkeypatch.setattr(cli, 'extract_scene_features', fail_extract)

        with pytest.raises(SystemExit) as exc_info:
            main(['sync', 'a.mp4', 'b.mp4', '--mode', 'scene'])
        assert exc_info.value.code == 2
        assert "pip install -e '.[scene]'" in capsys.readouterr().err


class TestMainDispatch:
    """Tests for main() routing between catalyst and scene paths."""

    def test_catalyst_pair_dispatches_run_sync_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(cli, 'run_sync_mode', lambda args: calls.append(args))
        monkeypatch.setattr(
            cli, 'run_scene_sync_mode',
            lambda args: pytest.fail("scene path taken in catalyst mode"),
        )

        main(['sync', 'a.mp4', 'b.mp4', '--mode', 'catalyst'])

        assert len(calls) == 1
        assert calls[0].mode == 'catalyst'

    def test_default_pair_dispatches_run_scene_sync_mode(self, monkeypatch):
        """With no --mode, sync now defaults to the scene path (T14)."""
        calls = []
        monkeypatch.setattr(
            cli, 'run_sync_mode',
            lambda args: pytest.fail("catalyst path taken by default"),
        )
        monkeypatch.setattr(
            cli, 'run_scene_sync_mode', lambda args: calls.append(args)
        )

        main(['sync', 'a.mp4', 'b.mp4'])

        assert len(calls) == 1
        assert calls[0].mode == 'scene'

    def test_scene_pair_dispatches_run_scene_sync_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli, 'run_sync_mode',
            lambda args: pytest.fail("catalyst path taken in scene mode"),
        )
        monkeypatch.setattr(
            cli, 'run_scene_sync_mode', lambda args: calls.append(args)
        )

        main(['sync', 'a.mp4', 'b.mp4', '--mode', 'scene'])

        assert len(calls) == 1
        # Defaults applied before dispatch
        assert calls[0].sample_hz == 10.0
        assert calls[0].matcher == 'aliked-lightglue'

    def test_scene_all_dispatches_run_scene_sync_all_mode(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            cli, 'run_sync_all_mode',
            lambda args: pytest.fail("catalyst batch path taken in scene mode"),
        )
        monkeypatch.setattr(
            cli, 'run_scene_sync_all_mode', lambda args: calls.append(args)
        )

        main(['sync', '--all', 'a.mp4', 'b.mp4', 'c.mp4', '--mode', 'scene'])

        assert len(calls) == 1
        assert calls[0].all == ['a.mp4', 'b.mp4', 'c.mp4']
        assert calls[0].embedder == 'dinov2-vitb14'

    def test_catalyst_all_rejects_scene_flags(self, monkeypatch, capsys):
        monkeypatch.setattr(
            cli, 'run_sync_all_mode',
            lambda args: pytest.fail("dispatch happened despite invalid flags"),
        )

        with pytest.raises(SystemExit) as exc_info:
            main(['sync', '--all', 'a.mp4', 'b.mp4', '--mode', 'catalyst', '--embedder', 'gist'])
        assert exc_info.value.code == 2
        assert '--embedder' in capsys.readouterr().err


class SyntheticEmptyMatcher:
    """Test-only matcher returning no matches (forces coarse fallback)."""

    name = "synthetic"

    def match(self, img_a, img_b, mask_a=None, mask_b=None):
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty


class TestEndToEndSceneSync:
    """End-to-end CLI invocation on tiny synthetic clips."""

    def test_scene_sync_produces_valid_csv(self, tmp_path, monkeypatch):
        # Two short clips with the same evolving noise pattern, B shifted 1 s
        video_a = create_synthetic_moving_noise_video(
            tmp_path / "vid_a", duration_s=5.0, fps=10.0, offset_s=0.0
        )
        video_b = create_synthetic_moving_noise_video(
            tmp_path / "vid_b", duration_s=5.0, fps=10.0, offset_s=1.0
        )
        if video_a is None or video_b is None:
            pytest.skip("mp4v codec unavailable")

        # Register the test-only synthetic matcher
        real_make_matcher = cli.make_matcher

        def make_matcher_with_synthetic(name):
            if name == "synthetic":
                return SyntheticEmptyMatcher()
            return real_make_matcher(name)

        monkeypatch.setattr(cli, 'make_matcher', make_matcher_with_synthetic)

        # Keep the embedding cache inside tmp_path
        real_extract = cli.extract_scene_features

        def extract_with_tmp_cache(video_path, embedder, sample_hz=10.0):
            return real_extract(
                video_path, embedder, sample_hz=sample_hz,
                cache_dir=tmp_path / "cache",
            )

        monkeypatch.setattr(cli, 'extract_scene_features', extract_with_tmp_cache)

        out_csv = tmp_path / "a_v_b_sync.csv"
        main([
            'sync', str(video_a), str(video_b),
            '--mode', 'scene',
            '--embedder', 'gist',
            '--matcher', 'synthetic',
            '--sample-hz', '5',
            '--max-sync-interval', '2.0',
            '-o', str(out_csv),
        ])

        assert out_csv.exists(), "CSV output file was not created"

        videos = read_csv(str(out_csv))
        assert len(videos) == 2

        for video in videos:
            timestamps = [seg.timestamp for seg in video.segments]
            assert len(timestamps) >= 2
            # Monotonic non-decreasing timestamps
            assert all(t1 <= t2 for t1, t2 in zip(timestamps, timestamps[1:]))

        # Sync points in A must respect the requested cadence
        times_a = [seg.timestamp for seg in videos[0].segments]
        gaps = np.diff(times_a)
        assert np.all(gaps <= 2.0 + 1e-6), f"Cadence violated: {gaps}"
