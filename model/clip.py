import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from torch import Tensor, nn
from torchaudio import transforms as T
from einops import pack, rearrange, unpack

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sample_rate: int = 48000,
        n_mel_channels: int = 80,
        center: bool = False,
        normalize: bool = True,
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
            mel_spectrogram = torch.pow(mel_spectrogram, 0.25)
        if self.normalize_log:
            mel_spectrogram = torch.log(torch.clamp(mel_spectrogram, min=1e-5))
        # Unpack non-spectrogram dimension
        return unpack(mel_spectrogram, ps, "* f l")[0]


class CLIP(nn.Module):
    def __init__(self, max_length=512):
        super(CLIP, self).__init__()
        self.configuration = CLIPConfig()
        self.max_length = max_length
        self.configuration.text_config.max_position_embeddings = max_length
        self.model = CLIPModel(self.configuration)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.mel = MelSpectrogram()

    def forward(self, audio, lyrics):
        # Mel spectrogram and stack stereo channels
        images = rearrange(self.mel(audio), "b c f l -> b (c f) l")
        # Turn the spectrograms to RGB for transfer learning
        images = images.unsqueeze(1).repeat(1, 3, 1, 1)

        inputs = self.processor(text=lyrics, images=images, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
        outputs = self.model(**inputs)

        batch_size = images.shape[0]
        labels = torch.arange(batch_size)
        loss_i = F.cross_entropy(outputs['logits_per_image'], labels) 
        loss_t = F.cross_entropy(outputs['logits_per_text'], labels)
        loss = (loss_i + loss_t)/2

        return loss
    
    @torch.no_grad()
    def encode_lyrics(self, lyrics):
        inputs = self.processor(text=lyrics, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True) 
        lyrics_features = self.model.get_text_features(**inputs)
        return lyrics_features

    def encode_audio(self, audio):
        # Mel spectrogram and stack stereo channels
        images = rearrange(self.mel(audio), "b c f l -> b (c f) l")
        # Turn the spectrograms to RGB for transfer learning
        images = images.unsqueeze(1).repeat(1, 3, 1, 1)

        inputs = self.processor(images=images, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
        audio_features = self.model.get_image_features(**inputs)
        return audio_features

    
if __name__ == "__main__":
    lyrics = """
    When I'm with you, I feel like myself 
    No stranger, the shadow of somebody else 
    When I feel you holdin' my hand 
    I get touched, ain't this life grand?

    But the form of a life is long never-ending 
    And the smell of your halo, I know 
    And the smile of a knife is seldom befriending 
    And the smell of tangelo, I know
    """
    clip = CLIP()
    clip.encode_lyrics(lyrics)

