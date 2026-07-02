"""Tests for the T13 validation harness.

Covers the synthetic viewpoint-perturbation validator (design SS8.2): a
synthetic clip and a cropped/scaled/rotated copy with a fake hood band must
scene-align to identity timing within one sample using the GistEmbedder.

Also smoke-tests scripts/validate_scene_alignment.py: the pure comparison
helpers (DeltaStats, mapping interpolation, OCR transition pairing), the
markdown log writer, and main() in both pair and --self modes with the heavy
pipelines monkeypatched (no real footage in CI).

Task reference: docs/scene_alignment_tasks.md T13
Design reference: docs/scene_alignment_design.md SS8
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tracksync.embedding import GistEmbedder
from tracksync.frame_analysis import FrameOCRData
from tracksync.models import SyncPoint, SyncResult
from tracksync.scene_align import coarse_align, extract_scene_features

from tests.test_scene_align_coarse import create_synthetic_scrolling_video

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts" / "validate_scene_alignment.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "validate_scene_alignment", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vsa = _load_script()


def create_synthetic_blob_video(
    output_dir: Path,
    duration_s: float,
    fps: float,
    width: int = 320,
    height: int = 240,
    keyframe_hz: float = 4.0,
    codec: str = "mp4v",
) -> Path | None:
    """Create a video of smoothly evolving low-frequency blob fields.

    Each frame linearly blends between random 9x12 keyframe fields (one per
    1/keyframe_hz seconds) upsampled to full resolution, so temporal identity
    is carried by globally distinct appearance rather than pattern position.
    Unlike a translating pattern (whose phase -- and hence apparent time --
    is shifted by cropping/rescaling), this content is robust to the
    crop/scale/rotate perturbation, making it suitable for the design SS8.2
    viewpoint-perturbation validator. A static hood band covers the bottom
    sixth of each frame.

    Returns:
        Path to the created video, or None if the codec is unavailable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "blobs.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*codec),
        fps, (width, height), isColor=True,
    )
    if not writer.isOpened():
        return None

    rng = np.random.default_rng(123)
    total_frames = int(duration_s * fps)
    n_keys = int(np.ceil(duration_s * keyframe_hz)) + 2
    keys = rng.uniform(0, 255, (n_keys, 3, 9, 12)).astype(np.float32)
    hood_rows = height // 6

    for frame_idx in range(total_frames):
        t = frame_idx / fps * keyframe_hz
        k = int(t)
        frac = t - k
        field = (1 - frac) * keys[k] + frac * keys[k + 1]  # [3, 9, 12]
        channels = [
            cv2.resize(field[c], (width, height),
                       interpolation=cv2.INTER_CUBIC)
            for c in range(3)
        ]
        frame = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        frame[height - hood_rows:, :] = (50, 50, 50)  # Static hood band
        writer.write(frame)

    writer.release()
    return video_path


def create_perturbed_copy(
    src_path: Path,
    output_path: Path,
    rotation_deg: float = 5.0,
    crop_frac: float = 0.90,
    hood_frac: float = 0.20,
) -> Path | None:
    """Create a viewpoint-perturbed copy of a video (design SS8.2).

    Per frame: rotate about the center, crop the central crop_frac region,
    scale back to the original resolution, and paint a fake static hood band
    over the bottom hood_frac of the frame.

    Returns:
        Path to the perturbed video, or None if the codec is unavailable.
    """
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return None

    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
    crop_w, crop_h = int(w * crop_frac), int(h * crop_frac)
    x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
    hood_rows = int(h * hood_frac)

    wrote_any = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.warpAffine(
            frame, rotation, (w, h), borderMode=cv2.BORDER_REFLECT
        )
        frame = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        frame[h - hood_rows:, :] = (30, 30, 35)  # Fake static hood band
        writer.write(frame)
        wrote_any = True

    cap.release()
    writer.release()
    return output_path if wrote_any else None


class TestSyntheticPerturbation:
    """Design SS8.2: perturbed copy must align to identity within 1 sample."""

    def test_perturbed_copy_aligns_to_identity(self, tmp_path):
        sample_hz = 5.0
        src = create_synthetic_blob_video(tmp_path, duration_s=6.0, fps=10.0)
        if src is None:
            pytest.skip("cv2.VideoWriter codec unavailable")
        perturbed = create_perturbed_copy(src, tmp_path / "perturbed.mp4")
        if perturbed is None:
            pytest.skip("cv2.VideoWriter codec unavailable")

        embedder = GistEmbedder()
        feat_a = extract_scene_features(
            str(src), embedder, sample_hz=sample_hz,
            cache_dir=tmp_path / "cache",
        )
        feat_b = extract_scene_features(
            str(perturbed), embedder, sample_hz=sample_hz,
            cache_dir=tmp_path / "cache",
        )

        # The fake hood band must register as static content
        assert feat_b.static_mask.any()

        coarse = coarse_align(feat_a, feat_b)

        tol = 1.0 / sample_hz
        t_lo = coarse.trim_start_a + tol
        t_hi = coarse.trim_end_a - tol
        assert t_hi > t_lo, "Trimmed overlap collapsed"
        grid = np.linspace(t_lo, t_hi, 50)
        mapped = np.array([coarse.f(float(t)) for t in grid])
        deviations = np.abs(mapped - grid)
        # 1 sample tolerance, plus 5% allowance for PCHIP smoothing overshoot
        # between DTW path points that are themselves within one sample.
        assert deviations.max() <= tol * 1.05, (
            f"Identity deviation {deviations.max():.3f}s exceeds "
            f"1 sample ({tol:.3f}s)"
        )

    def test_script_self_alignment_check_on_synthetic(self, tmp_path, monkeypatch):
        """Script's --self path (design SS8.1) on a synthetic clip."""
        sample_hz = 5.0
        src = create_synthetic_scrolling_video(
            tmp_path, duration_s=6.0, fps=10.0
        )
        if src is None:
            pytest.skip("cv2.VideoWriter codec unavailable")

        # Keep the embedding cache inside tmp_path
        def extract_with_tmp_cache(video_path, embedder, sample_hz=10.0):
            return extract_scene_features(
                video_path, embedder, sample_hz=sample_hz,
                cache_dir=tmp_path / "cache",
            )

        monkeypatch.setattr(
            vsa, 'extract_scene_features', extract_with_tmp_cache
        )

        max_dev, runtime = vsa.self_alignment_check(
            str(src), embedder_name="gist", sample_hz=sample_hz
        )

        assert max_dev <= 1.0 / sample_hz + 1e-6
        assert runtime > 0


class TestDeltaStats:
    """Tests for DeltaStats and format_stats."""

    def test_from_deltas_takes_abs(self):
        stats = vsa.DeltaStats.from_deltas(np.array([-1.0, 0.5, 2.0]))
        assert stats.max_dt == 2.0
        assert stats.n == 3
        assert stats.mean_dt == pytest.approx((1.0 + 0.5 + 2.0) / 3)

    def test_p95(self):
        stats = vsa.DeltaStats.from_deltas(np.linspace(0, 1, 101))
        assert stats.p95_dt == pytest.approx(0.95)

    def test_format_stats(self):
        stats = vsa.DeltaStats(max_dt=0.1, mean_dt=0.05, p95_dt=0.09, n=10)
        assert vsa.format_stats(stats) == "0.100 / 0.050 / 0.090"

    def test_format_stats_none(self):
        assert vsa.format_stats(None) == "n/a"


def _sync_result(points: list[tuple[float, float]]) -> SyncResult:
    sync_points = [
        SyncPoint(time_a=a, time_b=b, label="test") for a, b in points
    ]
    return SyncResult(
        sync_points=sync_points,
        trim_start_a=points[0][0] if points else 0.0,
        trim_end_a=points[-1][0] if points else 0.0,
        trim_start_b=points[0][1] if points else 0.0,
        trim_end_b=points[-1][1] if points else 0.0,
        crossings_a=[],
        crossings_b=[],
    )


class TestMappingComparison:
    """Tests for mapping_from_sync_result and compare_mappings."""

    def test_mapping_interpolates(self):
        result = _sync_result([(0.0, 10.0), (10.0, 25.0)])
        f, t_min, t_max = vsa.mapping_from_sync_result(result)
        assert (t_min, t_max) == (0.0, 10.0)
        assert f(5.0) == pytest.approx(17.5)

    def test_mapping_requires_two_points(self):
        with pytest.raises(ValueError):
            vsa.mapping_from_sync_result(_sync_result([(0.0, 0.0)]))

    def test_compare_constant_offset(self):
        identity = _sync_result([(0.0, 0.0), (10.0, 10.0)])
        offset = _sync_result([(0.0, 1.0), (10.0, 11.0)])
        stats = vsa.compare_mappings(offset, identity)
        assert stats is not None
        assert stats.max_dt == pytest.approx(1.0)
        assert stats.mean_dt == pytest.approx(1.0)
        assert stats.p95_dt == pytest.approx(1.0)

    def test_compare_identical_mappings(self):
        result = _sync_result([(0.0, 5.0), (4.0, 9.5), (10.0, 16.0)])
        stats = vsa.compare_mappings(result, result)
        assert stats.max_dt == pytest.approx(0.0)

    def test_compare_disjoint_domains_returns_none(self):
        early = _sync_result([(0.0, 0.0), (5.0, 5.0)])
        late = _sync_result([(20.0, 20.0), (30.0, 30.0)])
        assert vsa.compare_mappings(early, late) is None

    def test_compare_degenerate_result_returns_none(self):
        one_point = _sync_result([(0.0, 0.0)])
        ok = _sync_result([(0.0, 0.0), (10.0, 10.0)])
        assert vsa.compare_mappings(one_point, ok) is None


class TestOcrTransitionPairs:
    """Tests for OCR segment-transition extraction and pairing."""

    @staticmethod
    def _ocr(segments):
        return [FrameOCRData(segment_number=s) for s in segments]

    def test_pairs_matching_transitions(self):
        ocr_a = self._ocr([1, 1, 2, 2, 3, 3])
        times_a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        ocr_b = self._ocr([1, 2, 2, 3])
        times_b = [10.0, 11.0, 12.0, 13.0]

        pairs = vsa.ocr_transition_pairs(ocr_a, times_a, ocr_b, times_b)

        assert pairs == [(2.0, 11.0), (4.0, 13.0)]

    def test_repeated_transition_excluded(self):
        """A segment entered twice is lap-ambiguous and must be skipped."""
        ocr_a = self._ocr([1, 2, 1, 2, 3])
        times_a = [0.0, 1.0, 2.0, 3.0, 4.0]
        ocr_b = self._ocr([1, 2, 3])
        times_b = [0.0, 1.0, 2.0]

        pairs = vsa.ocr_transition_pairs(ocr_a, times_a, ocr_b, times_b)

        # Segment 2 entered twice in A -> only segment 3 pairs.
        # (Re-entering segment 1 also records it, but 1 has no transition in B.)
        assert pairs == [(4.0, 2.0)]

    def test_none_segments_skipped(self):
        ocr_a = [
            FrameOCRData(segment_number=1),
            None,
            FrameOCRData(segment_number=None),
            FrameOCRData(segment_number=2),
        ]
        times_a = [0.0, 1.0, 2.0, 3.0]
        ocr_b = self._ocr([1, 2])
        times_b = [0.0, 1.0]

        pairs = vsa.ocr_transition_pairs(ocr_a, times_a, ocr_b, times_b)

        assert pairs == [(3.0, 1.0)]

    def test_no_common_transitions(self):
        pairs = vsa.ocr_transition_pairs(
            self._ocr([1, 2]), [0.0, 1.0],
            self._ocr([5, 6]), [0.0, 1.0],
        )
        assert pairs == []

    def test_ocr_delta_stats(self):
        result = _sync_result([(0.0, 10.0), (10.0, 20.0)])  # f(t) = t + 10
        pairs = [(2.0, 12.5), (5.0, 15.0), (100.0, 200.0)]  # Last out of domain

        stats = vsa.ocr_delta_stats(result, pairs)

        assert stats.n == 2
        assert stats.max_dt == pytest.approx(0.5)

    def test_ocr_delta_stats_no_pairs(self):
        result = _sync_result([(0.0, 0.0), (10.0, 10.0)])
        assert vsa.ocr_delta_stats(result, []) is None


class TestMarkdownLog:
    """Tests for append_markdown_row."""

    def test_creates_template_when_missing(self, tmp_path):
        log = tmp_path / "log.md"
        vsa.append_markdown_row(log, "| 2026-01-01 | pair | x | 1 | 2 | a | b | c | notes |")

        content = log.read_text()
        assert content.startswith("# Scene Alignment Validation Results")
        assert "validate_scene_alignment.py" in content  # Operator commands
        assert "--self" in content
        assert content.rstrip().endswith("| notes |")

    def test_appends_to_existing(self, tmp_path):
        log = tmp_path / "log.md"
        vsa.append_markdown_row(log, "| row1 |")
        vsa.append_markdown_row(log, "| row2 |")

        lines = log.read_text().splitlines()
        assert lines[-2] == "| row1 |"
        assert lines[-1] == "| row2 |"

    def test_repo_template_matches_script_template(self):
        """docs/scene_alignment_validation.md must start with LOG_TEMPLATE.

        Validation runs append result rows after the template, so only the
        template prefix is compared.
        """
        repo_log = SCRIPT_PATH.parent.parent / "docs" / "scene_alignment_validation.md"
        assert repo_log.read_text().startswith(vsa.LOG_TEMPLATE)


class TestMainSmoke:
    """Smoke tests for main() with the heavy pipelines monkeypatched."""

    def _fake_features(self, segments, times):
        return SimpleNamespace(
            interpolated_ocr=[FrameOCRData(segment_number=s) for s in segments],
            frame_times=times,
        )

    def test_pair_mode_end_to_end(self, tmp_path, monkeypatch, capsys):
        catalyst_result = _sync_result([(0.0, 1.0), (10.0, 11.0)])
        scene_result = _sync_result([(0.0, 1.2), (10.0, 11.2)])
        features_a = self._fake_features([1, 2, 3], [0.0, 4.0, 8.0])
        features_b = self._fake_features([1, 2, 3], [1.0, 5.0, 9.0])

        monkeypatch.setattr(
            vsa, 'run_catalyst_sync',
            lambda va, vb, max_sync_interval: (
                catalyst_result, features_a, features_b, 1.25
            ),
        )
        monkeypatch.setattr(
            vsa, 'run_scene_sync',
            lambda va, vb, **kw: (scene_result, 4.5),
        )

        log = tmp_path / "log.md"
        vsa.main([
            'a.mp4', 'b.mp4',
            '--embedder', 'gist', '--matcher', 'fake',
            '--log', str(log),
        ])

        out = capsys.readouterr().out
        assert "Scene vs Catalyst" in out
        assert "0.200" in out  # Constant 0.2 s method disagreement

        content = log.read_text()
        row = content.rstrip().splitlines()[-1]
        assert "| pair |" in row
        assert "a.mp4 vs b.mp4" in row
        assert "1.2 | 4.5" in row  # Runtimes
        assert "ocr_pairs=2" in row  # Transitions into segments 2 and 3

    def test_self_mode_end_to_end(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            vsa, 'self_alignment_check',
            lambda video, embedder_name, sample_hz, band_pct: (0.05, 2.0),
        )

        log = tmp_path / "log.md"
        vsa.main(['--self', 'v.mp4', '--embedder', 'gist', '--log', str(log)])

        out = capsys.readouterr().out
        assert "SELF-ALIGNMENT CHECK" in out
        assert "0.050" in out

        row = log.read_text().rstrip().splitlines()[-1]
        assert "| self |" in row
        assert "v.mp4" in row
        assert "0.050" in row

    def test_requires_exactly_two_videos(self):
        with pytest.raises(SystemExit) as exc_info:
            vsa.main(['a.mp4'])
        assert exc_info.value.code == 2

    def test_self_rejects_positional_videos(self):
        with pytest.raises(SystemExit) as exc_info:
            vsa.main(['--self', 'v.mp4', 'a.mp4', 'b.mp4'])
        assert exc_info.value.code == 2

    def test_torch_backends_exit_2_when_deps_missing(self, monkeypatch, capsys):
        from tracksync.scene_deps import MissingSceneDependenciesError

        def mock_require():
            raise MissingSceneDependenciesError()

        monkeypatch.setattr(vsa, 'require_scene_deps', mock_require)
        monkeypatch.setattr(
            vsa, 'run_catalyst_sync',
            lambda *a, **k: pytest.fail("pipeline ran despite missing deps"),
        )

        with pytest.raises(SystemExit) as exc_info:
            vsa.main(['a.mp4', 'b.mp4'])  # Default embedder needs torch
        assert exc_info.value.code == 2
        assert "pip install -e '.[scene]'" in capsys.readouterr().err

    def test_gist_skips_dep_check(self, monkeypatch):
        def mock_require():
            raise AssertionError("require_scene_deps must not be called")

        monkeypatch.setattr(vsa, 'require_scene_deps', mock_require)
        vsa.check_scene_deps('gist', 'fake')  # Must not raise
