import os
import signal

import comet_ml
import lightning as L
import torch
import torch.nn.functional as F
from comet_ml import Experiment
from data.dataset import get_dataset
from einops import pack, rearrange, unpack
from ema import EMA
from lightning import Trainer
from lightning.pytorch.callbacks import (GradientAccumulationScheduler,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel,
                          CLIPTokenizer)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class CLIP(L.LightningModule):
    def __init__(self, max_length=512, crop=2**20, batch_size=256, dataset_path=None, num_workers=16):
        super(CLIP, self).__init__()
        self.save_hyperparameters()
        self.configuration = CLIPConfig()
        self.max_length = max_length
        self.crop = crop
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.configuration.text_config.max_position_embeddings = max_length
        self.model = CLIPModel(self.configuration)
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.mel = MelSpectrogram()
       

    def training_step(self, batch, batch_idx):
        audio, *_, lyrics = batch
        # Mel spectrogram and stack stereo channels
        images = rearrange(self.mel(audio.to(device)), "b c f l -> b (c f) l")
        # Turn the spectrograms to RGB for transfer learning
        images = images.unsqueeze(1).repeat(1, 3, 1, 1).to(device)

        images_processed = self.processor(images=images, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).to(device)
        lyrics_tokenized = self.tokenizer(text=lyrics, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).to(device)
        inputs = {**images_processed, **lyrics_tokenized}
        outputs = self.model(**inputs)

        batch_size = images.shape[0]
        labels = torch.arange(batch_size).to(device)
        loss_i = F.cross_entropy(outputs['logits_per_image'], labels) 
        loss_t = F.cross_entropy(outputs['logits_per_text'], labels)
        loss = (loss_i + loss_t)/2
        self.log('loss_img', loss_i, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('loss_txt', loss_t, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)
    
    @torch.no_grad()
    def encode_lyrics(self, lyrics):
        lyrics_tokenized = self.tokenizer(text=lyrics, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).to(device)
        lyrics_features = self.model.get_text_features(**lyrics_tokenized).to(device)
        return lyrics_features

    @torch.no_grad()
    def encode_audio(self, audio):
        # Mel spectrogram and stack stereo channels
        images = rearrange(self.mel(audio.to(device)), "b c f l -> b (c f) l")
        # Turn the spectrograms to RGB for transfer learning
        images = images.unsqueeze(1).repeat(1, 3, 1, 1).to(device)

        images_processed = self.processor(images=images, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True).to(device)
        audio_features = self.model.get_image_features(**images_processed).to(device)
        return audio_features
    
    def train_dataloader(self):
        dataset = get_dataset(self.dataset_path, crop=self.crop)
        train_loader = DataLoader(dataset, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, batch_size=self.batch_size, shuffle=True)
        return train_loader

    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--precision", type=str, default='16-mixed')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--default_root_dir", type=str, default="/home/gconcialdi/ainur/runs/")
    parser.add_argument("--gradient_clip", type=float, default=1)
    parser.add_argument("--sanity_steps", type=int, default=0)


    # Hyperparameters for the model
    parser.add_argument("--dataset_path", type=str, default="/home/gconcialdi/spotdl/")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--crop", type=int, default=2**20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)



    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    
    logger = CometLogger(
        api_key="9LmOAqSG4omncUN3QT42iQoqb",
        project_name="ainur",
        workspace="gio99c",
        experiment_name="clip",
        offline=False
        )


    clip = CLIP(max_length=args.max_length, crop=args.crop, batch_size=args.batch_size, dataset_path=args.dataset_path, num_workers=args.num_workers)
    accumulator = GradientAccumulationScheduler(scheduling={0: 4, 600: 2, 900: 1})
    ema = EMA(0.999)

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.default_root_dir, "clip/checkpoints/"))
    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      precision=args.precision,
                      accelerator=args.accelerator,
                      devices=args.n_devices,
                      num_nodes=args.num_nodes,
                      default_root_dir=args.default_root_dir,
                      num_sanity_val_steps=args.sanity_steps,
                      gradient_clip_val=args.gradient_clip,
                      plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
                      callbacks=[StochasticWeightAveraging(swa_lrs=1e-4), checkpoint_callback, accumulator, ema])

    trainer.fit(clip, ckpt_path=args.checkpoint_path)
