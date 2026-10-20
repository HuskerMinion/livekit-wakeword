"""Feature extraction: audio → mel-spectrogram → speech embeddings → .npy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from onnxruntime.capi.onnxruntime_pybind11_state import SessionOptions

from ..config import WakeWordConfig
from ..models.feature_extractor import MelSpectrogramFrontend, SpeechEmbedding
from ..resources import get_embedding_model_path, get_mel_model_path

logger = logging.getLogger(__name__)

# Target: 16 embedding timesteps per training example
N_EMBEDDING_TIMESTEPS = 16
# Speech-embedding input: 76 mel frames per window, 8 frames between windows
_WINDOW, _STRIDE = 76, 8


def _pad_or_truncate(embeddings: np.ndarray) -> np.ndarray:
    """Take last N_EMBEDDING_TIMESTEPS or left-pad a (n_windows, 96) embedding."""
    if embeddings.shape[0] >= N_EMBEDDING_TIMESTEPS:
        return embeddings[-N_EMBEDDING_TIMESTEPS:]
    pad = np.zeros(
        (N_EMBEDDING_TIMESTEPS - embeddings.shape[0], 96),
        dtype=np.float32,
    )
    return np.concatenate([pad, embeddings], axis=0)


def extract_features_from_directory(
    clip_dir: Path,
    mel_frontend: MelSpectrogramFrontend,
    speech_embedding: SpeechEmbedding,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract (N_clips, 16, 96) features from a directory of WAV files.

    Processes clips through MelSpectrogramFrontend → SpeechEmbedding,
    then takes last 16 embedding timesteps per clip. Only those last
    windows are computed, and the windows of ``batch_size`` clips go
    through the embedding model in one call.
    """
    import re

    import soundfile as sf
    from tqdm import tqdm

    # Only process augmented clips (_rN.wav), skip clean TTS originals
    _aug_re = re.compile(r"^clip_\d{6}_r\d+\.wav$")
    wav_files = sorted(p for p in clip_dir.glob("*.wav") if _aug_re.match(p.name))
    if not wav_files:
        logger.warning(f"No WAV files in {clip_dir}")
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    all_features: list[np.ndarray] = []

    with tqdm(total=len(wav_files), desc=f"Features {clip_dir.name}", unit="clip") as pbar:
        for start in range(0, len(wav_files), batch_size):
            # Mel windows (76 frames, hop 8) per clip, keeping only the last 16
            window_sets: list[np.ndarray] = []
            for wav_path in wav_files[start : start + batch_size]:
                audio, sr = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio[:, 0]
                mel = mel_frontend(audio.astype(np.float32))[0]  # (time_frames, 32)
                n_windows = max(0, (mel.shape[0] - _WINDOW) // _STRIDE + 1)
                first = max(0, n_windows - N_EMBEDDING_TIMESTEPS)
                window_sets.append(
                    np.stack(
                        [mel[i * _STRIDE : i * _STRIDE + _WINDOW] for i in range(first, n_windows)]
                    )
                    if n_windows
                    else np.zeros((0, _WINDOW, 32), dtype=np.float32)
                )

            windows = np.concatenate(window_sets)
            embeddings = (
                speech_embedding(windows) if len(windows) else np.zeros((0, 96), np.float32)
            )
            offset = 0
            for ws in window_sets:
                all_features.append(_pad_or_truncate(embeddings[offset : offset + len(ws)]))
                offset += len(ws)
            pbar.update(len(window_sets))

    if not all_features:
        return np.zeros((0, N_EMBEDDING_TIMESTEPS, 96), dtype=np.float32)

    return np.stack(all_features, axis=0)  # (N_clips, 16, 96)


def run_extraction(config: WakeWordConfig, sess_options: SessionOptions | None = None) -> None:
    """Extract and save features for all splits of a wake word config."""
    mel_frontend = MelSpectrogramFrontend(
        onnx_path=get_mel_model_path(),
        sess_options=sess_options,
    )
    speech_embedding = SpeechEmbedding(
        onnx_path=get_embedding_model_path(),
        sess_options=sess_options,
    )

    model_dir = config.model_output_dir
    splits = [
        ("positive_train", "positive_features_train.npy"),
        ("positive_test", "positive_features_test.npy"),
        ("negative_train", "negative_features_train.npy"),
        ("negative_test", "negative_features_test.npy"),
        ("background_train", "background_noise_features_train.npy"),
        ("background_test", "background_noise_features_test.npy"),
    ]

    for clip_subdir, feature_filename in splits:
        clip_dir = model_dir / clip_subdir
        if not clip_dir.exists():
            logger.warning(f"Skipping feature extraction for {clip_subdir}: not found")
            continue

        logger.info(f"Extracting features from {clip_dir}...")
        features = extract_features_from_directory(
            clip_dir=clip_dir,
            mel_frontend=mel_frontend,
            speech_embedding=speech_embedding,
        )

        out_path = model_dir / feature_filename
        np.save(str(out_path), features)
        logger.info(f"Saved {features.shape} features to {out_path}")
