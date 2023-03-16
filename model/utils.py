import torch
from torch import Tensor, nn
from torchaudio import transforms as T
from einops import pack, rearrange, reduce, unpack
import torch.nn.functional as F
from typing import Any, Callable, Optional, Sequence, Type, TypeVar, Union

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sample_rate: int = 48000,
        n_mel_channels: int = 80,
        center: bool = False,
        normalize: bool = False,
        normalize_log: bool = False,
    ):
        super().__init__()
        self.padding = (n_fft - hop_length) // 2
        self.normalize = normalize
        self.normalize_log = normalize_log
        self.hop_length = hop_length

        self.to_spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=center,
            power=None,
        )

        self.to_mel_scale = T.MelScale(
            n_mels=n_mel_channels, n_stft=n_fft // 2 + 1, sample_rate=sample_rate
        )

    def forward(self, waveform: Tensor) -> Tensor:
        # Pack non-time dimension
        waveform, ps = pack([waveform], "* t")
        # Pad waveform
        waveform = F.pad(waveform, [self.padding] * 2, mode="reflect")
        # Compute STFT
        spectrogram = self.to_spectrogram(waveform)
        # Compute magnitude
        spectrogram = torch.abs(spectrogram)
        # Convert to mel scale
        mel_spectrogram = self.to_mel_scale(spectrogram)
        # Normalize
        if self.normalize:
            mel_spectrogram = mel_spectrogram / torch.max(mel_spectrogram)
            mel_spectrogram = 2 * torch.pow(mel_spectrogram, 0.25) - 1
        if self.normalize_log:
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        # Unpack non-spectrogram dimension
        return unpack(mel_spectrogram, ps, "* f l")[0]