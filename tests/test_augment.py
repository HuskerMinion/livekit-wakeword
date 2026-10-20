"""Tests for augmentation utilities."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import soundfile as sf

from livekit.wakeword.data.augment import AudioAugmentor, _augment_directory, align_clip_to_end


class TestAlignClipToEnd:
    def test_basic_alignment(self):
        audio = np.ones(8000, dtype=np.float32)  # 0.5s at 16kHz
        target_length = 32000  # 2s
        result = align_clip_to_end(audio, target_length, jitter_samples=0)
        assert result.shape == (target_length,)
        # Audio should be at the end
        assert np.sum(result[-8000:]) > 0
        assert np.sum(result[:16000]) == 0.0

    def test_output_length(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000

    def test_longer_clip_than_target(self):
        audio = np.ones(48000, dtype=np.float32)  # 3s, longer than target
        result = align_clip_to_end(audio, 32000, jitter_samples=0)
        assert len(result) == 32000


class TestAugmentDirectory:
    @pytest.fixture
    def clip_dir(self, tmp_path):
        """Eight identical clips plus a background noise file."""
        rng = np.random.default_rng(0)
        clips = tmp_path / "positive_train"
        clips.mkdir()
        speech = (0.3 * rng.standard_normal(12000)).astype(np.float32)
        for i in range(8):
            sf.write(clips / f"clip_{i:06d}.wav", speech, 16000)
        bg = tmp_path / "backgrounds"
        bg.mkdir()
        sf.write(bg / "noise.wav", (0.1 * rng.standard_normal(64000)).astype(np.float32), 16000)
        return clips, AudioAugmentor(background_paths=[bg], rir_paths=[])

    @pytest.mark.parametrize("workers", [1, 3])
    def test_writes_every_clip(self, clip_dir, workers):
        clips, augmentor = clip_dir
        _augment_directory(clips, augmentor, is_positive=True, num_workers=workers)
        outs = sorted(clips.glob("clip_*_r0.wav"))
        assert len(outs) == 8
        assert all(sf.info(str(o)).frames == 32000 for o in outs)

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parallel path needs fork")
    def test_workers_draw_different_augmentations(self, tmp_path):
        # Forked workers inherit the parent's numpy RNG state (Python's `random` is
        # reseeded at fork, numpy is not); without reseeding, workers would apply
        # the same numpy-drawn augmentation sequence to their clips.
        class NumpyNoise(AudioAugmentor):
            def augment_clip(self, audio):
                return audio + 0.01 * np.random.standard_normal(len(audio)).astype(np.float32)

        clips = tmp_path / "negative_train"
        clips.mkdir()
        for i in range(16):
            sf.write(clips / f"clip_{i:06d}.wav", np.zeros(32000, np.float32), 16000)
        _augment_directory(clips, NumpyNoise([], []), is_positive=False, num_workers=4)
        outs = [sf.read(str(o))[0].tobytes() for o in sorted(clips.glob("clip_*_r0.wav"))]
        assert len(set(outs)) == len(outs)
