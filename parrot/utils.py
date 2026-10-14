"""Utility Functions for PARROT"""

import torch
import numpy as np
import librosa
import soundfile as sf


def extract_speech_features(audio_path: str, sr: int = 16000) -> torch.Tensor:
    """Extract speech features from audio file"""
    waveform, _ = librosa.load(audio_path, sr=sr)
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    return waveform


def load_audio(audio_path: str, sr: int = 16000):
    """Load audio file"""
    audio, sr_orig = librosa.load(audio_path, sr=sr)
    return torch.from_numpy(audio).float(), sr


def save_audio(audio: torch.Tensor, path: str, sr: int = 16000):
    """Save audio to file"""
    if audio.is_cuda:
        audio = audio.cpu()
    audio = audio.numpy()
    sf.write(path, audio, sr)


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)


EMOTIONS = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Surprised",
    5: "Disgusted",
    6: "Fearful",
}


def emotion_id_to_label(emotion_id: int) -> str:
    """Convert emotion ID to label"""
    return EMOTIONS.get(emotion_id, "Unknown")
