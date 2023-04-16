import os
import signal

import comet_ml
import lightning as L
import numpy as np
import torch
import torchaudio
from audio_diffusion_pytorch import (DiffusionModel, UNetV0, VDiffusion,
                                     VSampler)
from autoencoder import LitDAE
from clip import CLIP
from data.dataset import get_dataset
from ema import EMA
from fad import FAD
from lightning import Trainer
from lightning.pytorch.callbacks import (GradientAccumulationScheduler,
                                         ModelCheckpoint,
                                         StochasticWeightAveraging)
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader, random_split


def select(key, **kwargs):
    value = kwargs.get(key, None)
    return value

class Ainur(L.LightningModule):
    def __init__(self, 
                 inject_depth, 
                 dataset_path, 
                 crop=2**20, 
                 in_channels=32, 
                 channels=[128, 256, 512, 512, 1024, 1024], 
                 num_workers=16, 
                 batch_size=64, 
                 sample_length=2**20, 
                 latent_factor=9, 
                 clip_checkpoint_path=".",
                 num_steps=50,
                 embedding_scale=5.0,
                 checkpoint_every_n_epoch=10):
        super(Ainur, self).__init__()
        self.save_hyperparameters()
        self.inject_depth = inject_depth
        self.dataset = get_dataset(dataset_path, crop=crop)
        self.dataset_path = dataset_path
        self.crop = crop
        self.in_channels = in_channels
        self.channels = channels
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.latent_factor = latent_factor
        self.sample_length = sample_length
        self.num_steps = num_steps
        self.embedding_scale = embedding_scale
        self.checkpoint_every_n_epoch = checkpoint_every_n_epoch
        self.frechet_lyrics = FAD()
        self.frechet_audio = FAD()
        self.frechet_noclip = FAD()
        self.clip = CLIP.load_from_checkpoint(clip_checkpoint_path)
        self.clip.eval()
        self.autoencoder = LitDAE(dataset_path)
        self.autoencoder.eval()



        context_channels = [0] * len(channels)
        context_channels[inject_depth] = 1 ## encoder.out_channels
        self.diffusion_model = DiffusionModel(
            net_t=UNetV0,
            in_channels=in_channels, # U-Net: number of input/output (audio) channels
            channels=channels, # U-Net: channels at each layer
            factors=[1, 2, 2, 2, 2, 2], # U-Net: downsampling and upsampling factors at each layer
            items=[2, 2, 2, 4, 8, 8], # U-Net: number of repeating items at each layer
            attentions=[0, 0, 1, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
            attention_heads=12, # U-Net: number of attention heads per attention item
            attention_features=64, # U-Net: number of attention features per attention item
            use_text_conditioning=True, # U-Net: enables text conditioning (default T5-base)
            use_embedding_cfg=True, # U-Net: enables classifier free guidance
            embedding_max_length=64, # U-Net: text embedding maximum length (default for T5-base)
            embedding_features=768, # U-Net: text embedding features (default for T5-base)
            cross_attentions=[1, 1, 1, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer
            context_channels=context_channels, # important to inject the clip embeddings
            diffusion_t=VDiffusion, 
            sampler_t=VSampler)
        

    def training_step(self, batch, batch_idx, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        audio, text, _ = batch
        audio = audio.to(device)
        #latent = torch.cat((self.clip.encode_lyrics(lyrics).unsqueeze(1), self.clip.encode_audio(audio).unsqueeze(1)), dim=1) # b x 2 x 512
        latent = self.clip.encode_audio(audio).unsqueeze(1).to(device) # b x 1 x 512
        channels = [None] * self.inject_depth + [latent]
        encoded_audio = self.autoencoder.encode(audio).to(device)
        
        # Compute diffusion loss
        batch_size = audio.shape[0]
        loss = self.diffusion_model(encoded_audio, 
                                    text=text, 
                                    channels=channels, 
                                    embedding_mask_proba=0.1,
                                    **kwargs)
        with torch.no_grad():
            # Log loss
            self.log('loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.checkpoint_every_n_epoch == 0:
            audio, text, lyrics = batch
            audio = audio.cpu()
            text = text
            lyrics = lyrics

            # Create hidden tmp directory
            tmp_dir = os.path.join(os.getcwd(), ".tmp")
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)

            # Check if the statistics are already computed
            if not os.path.exists(os.path.join(tmp_dir, "background_statistics.ptc")):
                train_dataset , *_ = random_split(self.dataset, [0.98, 0.005, 0.015], torch.Generator().manual_seed(42))
                background, _ = random_split(train_dataset, [0.1, 0.9], torch.Generator().manual_seed(42))
                background = map(lambda item : item[0], background)
            else:
                background = None
            
            with torch.no_grad():
                # Log original audio 
                torchaudio.save(os.path.join(tmp_dir, f"original_{batch_idx}.wav"), audio[0].detach().cpu(), 48_000) 
                self.logger.experiment.log_audio(os.path.join(tmp_dir, f"original_{batch_idx}.wav"))

                # Compute fad and log audio
                self.evaluate(text, lyrics, mode='lyrics', lyrics=lyrics, background=background, batch_idx=batch_idx)
                self.evaluate(text, audio, mode='audio', background=background, batch_idx=batch_idx)
                self.evaluate(text, mode='noclip', background=background, batch_idx=batch_idx)

    def on_validation_epoch_end(self):
        if self.current_epoch % self.checkpoint_every_n_epoch == 0:
            self.log("FAD_lyrics", self.frechet_lyrics.compute(), on_epoch=True, prog_bar=True)
            self.log("FAD_audio", self.frechet_audio.compute(), on_epoch=True, prog_bar=True)
            self.log("FAD_noclip", self.frechet_noclip.compute(), on_epoch=True, prog_bar=True)
            self.frechet_lyrics.reset()
            self.frechet_audio.reset()
            self.frechet_noclip.reset()


    def test_step(self, batch, batch_idx):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        audio, text, lyrics = batch
        audio = audio.cpu()
        text = text
        lyrics = lyrics
        batch_size = audio.shape[0]

        # Create hidden tmp directory
        tmp_dir = os.path.join(os.getcwd(), ".tmp")
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

        # Check if the statistics are already computed
        if not os.path.exists(os.path.join(tmp_dir, "background_statistics.ptc")):
            train_dataset , *_ = random_split(self.dataset, [0.98, 0.005, 0.015], torch.Generator().manual_seed(42))
            background, _ = random_split(train_dataset, [0.1, 0.9], torch.Generator().manual_seed(42))
            background = map(lambda item : item[0], background)
        else:
            background = None
        
        with torch.no_grad():
            # Log original audio 
            torchaudio.save(os.path.join(tmp_dir, f"original_{batch_idx}.wav"), audio[0].detach().cpu(), 48_000) 
            self.logger.experiment.log_audio(os.path.join(tmp_dir, f"original_{batch_idx}.wav"))

            # Compute fad and log audio
            self.evaluate(text, lyrics, mode='lyrics', background=background, batch_size=batch_size, batch_idx=batch_idx)
            self.evaluate(text, audio, mode='audio', background=background, batch_size=batch_size, batch_idx=batch_idx)
            self.evaluate(text, mode='noclip', background=background, batch_size=batch_size, batch_idx=batch_idx)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.diffusion_model.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)
    
    def test_dataloader(self):
        *_, test_dataset = random_split(self.dataset, [0.98, 0.005, 0.015], torch.Generator().manual_seed(42))
        test_loader = DataLoader(test_dataset, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, batch_size=self.batch_size, shuffle=False)
        return test_loader
    
    def val_dataloader(self):
        _, val_dataset, _ = random_split(self.dataset, [0.98, 0.005, 0.015], torch.Generator().manual_seed(42))
        val_loader = DataLoader(val_dataset, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, batch_size=self.batch_size, shuffle=False)
        return val_loader
    
    def train_dataloader(self):
        train_dataset, *_ = random_split(self.dataset, [0.98, 0.005, 0.015], torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_dataset, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, batch_size=self.batch_size, shuffle=True)
        return train_loader
    

    @torch.no_grad()
    def evaluate(self, text, latent=None, mode='lyrics', lyrics=None, background=None, test=False, tmp_dir=".tmp", batch_idx=None):
        if mode == 'lyrics':
            evaluation = self.sample_audio(lyrics=latent, text=text, embedding_scale=self.embedding_scale, num_steps=self.num_steps).cpu()
            self.frechet_lyrics(evaluation, target=background)
        elif mode == 'audio':
            evaluation = self.sample_audio(audio=latent, text=text, embedding_scale=self.embedding_scale, num_steps=self.num_steps).cpu()
            self.frechet_audio(evaluation, target=background)
        elif (latent is None) and (mode == 'noclip'):
            evaluation = self.sample_audio(n_samples=len(text), text=text, embedding_scale=self.embedding_scale, num_steps=self.num_steps).cpu()
            self.frechet_noclip(evaluation, target=background)
        else:
            print(f"Unknown mode='{mode}', expected one of 'lyrics', 'audio', 'noclip'.")
            return -1

        torchaudio.save(os.path.join(tmp_dir, f"sample_{mode}{'_test' if test else ''}{f'_{batch_idx}' if batch_idx is not None else ''}.wav"), 
                                     evaluation[0].detach().cpu(), 
                                     48_000)
        self.logger.experiment.log_audio(os.path.join(tmp_dir, 
                                                      f"sample_{mode}{'_test' if test else ''}{f'_{batch_idx}' if batch_idx is not None else ''}.wav"))
        self.logger.experiment.log_text(f"{f'batch_idx={batch_idx}_' if batch_idx is not None else ''}{text[0]}{f'_lyrics: {lyrics[0]}' if lyrics is not None else ''}")
        del evaluation
        

    @torch.no_grad()
    def sample(self, lyrics=None, audio=None, n_samples=1, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if (lyrics is not None) or (audio is not None):
            n_samples = len(lyrics) if lyrics else audio.shape[0]

        if lyrics is not None:
            latent = self.clip.encode_lyrics(lyrics).unsqueeze(1).to(device) # conditioning on lyrics
        elif audio is not None:
            latent = self.clip.encode_audio(audio).unsqueeze(1).to(device) # conditioning on audio
        else:
            latent = torch.zeros(n_samples, 1, 2**self.latent_factor).to(device)  # no clip conditioning

        # Create the noise tensor
        noise = torch.randn(n_samples, self.in_channels, self.sample_length // 2**self.latent_factor).to(device)

        # Compute context from lyrics embedding
        channels = [None] * self.inject_depth + [latent]

        # Decode by sampling while conditioning on latent channels
        return self.diffusion_model.sample(noise, channels=channels, **kwargs).to(device)
    

    @torch.no_grad()
    def sample_audio(self, lyrics=None, audio=None, n_samples=1, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        samples = self.sample(lyrics=lyrics, audio=audio, n_samples=n_samples, **kwargs).to(device)
        num_steps = select("num_steps", **kwargs)

        return self.autoencoder.decode(samples, num_steps=num_steps).to(device)


        
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
    parser.add_argument("--clip_checkpoint_path", type=str, default="/home/gconcialdi/ainur/runs/clip/checkpoints/clip.ckpt")
    parser.add_argument("--default_root_dir", type=str, default="/home/gconcialdi/ainur/runs/")
    parser.add_argument("--checkpoint_every_n_epoch", type=int, default=10)
    parser.add_argument("--gradient_clip", type=float, default=0.25)


    # Hyperparameters for the model
    parser.add_argument("--dataset_path", type=str, default="/home/gconcialdi/spotdl/")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--crop", type=int, default=2**20)
    parser.add_argument("--sample_length", type=int, default=2**20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--embedding_scale", type=float, default=7.0)


    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    
    logger = CometLogger(
        api_key="9LmOAqSG4omncUN3QT42iQoqb",
        project_name="ainur",
        workspace="gio99c",
        experiment_name="ainur_v2",
        offline=False
        )

    inject_depth = int(np.log2(args.crop / 2**18))
    ainur = Ainur(inject_depth=inject_depth, 
                  crop=args.crop, 
                  dataset_path=args.dataset_path, 
                  num_workers=args.num_workers, 
                  batch_size=args.batch_size, 
                  clip_checkpoint_path=args.clip_checkpoint_path,
                  sample_length=args.sample_length,
                  num_steps=args.num_steps,
                  embedding_scale=args.embedding_scale,
                  checkpoint_every_n_epoch=args.checkpoint_every_n_epoch
                  )

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.default_root_dir, "ainur_model_v2/checkpoints/"), monitor="FAD_lyrics", save_last=True)
    accumulator = GradientAccumulationScheduler(scheduling={0: 4, 300: 2, 500: 1})
    ema = EMA(0.995)
    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      precision=args.precision,
                      accelerator=args.accelerator,
                      devices=args.n_devices,
                      num_nodes=args.num_nodes,
                      default_root_dir=args.default_root_dir,
                      num_sanity_val_steps=0,
                      gradient_clip_val=args.gradient_clip,
                      plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
                      callbacks=[StochasticWeightAveraging(swa_lrs=1e-4), checkpoint_callback, accumulator, ema])

    trainer.fit(ainur, ckpt_path=args.checkpoint_path)