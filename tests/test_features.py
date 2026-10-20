"""Tests for directory feature extraction."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from livekit.wakeword.data.features import (
    N_EMBEDDING_TIMESTEPS,
    _pad_or_truncate,
    extract_features_from_directory,
)
from livekit.wakeword.models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from livekit.wakeword.resources import get_embedding_model_path, get_mel_model_path


@pytest.fixture(scope="module")
def models():
    return (
        MelSpectrogramFrontend(onnx_path=get_mel_model_path()),
        SpeechEmbedding(onnx_path=get_embedding_model_path()),
    )


def _reference(path, mel_frontend, speech_embedding):
    """Unbatched path: every window of the clip, then the last 16."""
    audio, _ = sf.read(str(path))
    mel = mel_frontend(audio.astype(np.float32))
    return _pad_or_truncate(speech_embedding.extract_embeddings(mel)[0])


@pytest.mark.parametrize("batch_size", [1, 2, 256])
def test_batched_matches_per_clip(tmp_path, models, batch_size):
    rng = np.random.default_rng(0)
    # 2 s training clips, a longer one, and clips too short for 16 (or any) windows
    for i, seconds in enumerate([2.0, 2.0, 3.5, 1.0, 0.5]):
        audio = 0.1 * rng.standard_normal(int(seconds * 16000))
        sf.write(tmp_path / f"clip_{i:06d}_r0.wav", audio.astype(np.float32), 16000)

    feats = extract_features_from_directory(tmp_path, *models, batch_size=batch_size)

    assert feats.shape == (5, N_EMBEDDING_TIMESTEPS, 96)
    for i, path in enumerate(sorted(tmp_path.glob("*.wav"))):
        np.testing.assert_allclose(feats[i], _reference(path, *models), rtol=1e-4, atol=1e-5)
