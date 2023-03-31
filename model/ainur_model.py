import comet_ml
import lightning as L
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from clip import CLIP
import torch
from data.dataset import get_dataset
from torch.utils.data import DataLoader
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader
import signal
from autoencoder import LitDAE


class Ainur(L.LightningModule):
    def __init__(self, inject_depth, dataset_path, crop=2**18, in_channels=32, channels=[128, 256, 512, 512, 1024, 1024], num_workers=16, batch_size=64, latent_factor = None, adapter = None):
        super(Ainur, self).__init__()
        self.save_hyperparameters()
        self.inject_depth = inject_depth
        self.dataset = get_dataset(dataset_path, crop=crop)
        self.in_channels = in_channels
        self.channels = channels
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.clip = CLIP.load_from_checkpoint("/Users/gio/Desktop/checkpoint.ckpt")
        self.clip.eval()
        self.autoencoder = LitDAE(dataset_path)
        self.autoencoder.eval()


        # Optional custom latent factor and adapter
        #self.latent_factor = default(latent_factor, self.encoder.downsample_factor)
        #self.adapter = adapter.requires_grad_(False) if exists(adapter) else None

        context_channels = [0] * len(channels)
        context_channels[inject_depth] = 2 ## encoder.out_channels
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
        audio, text, lyrics = batch
        latent = torch.cat((self.clip.encode_lyrics(lyrics).unsqueeze(1), self.clip.encode_audio(audio).unsqueeze(1)), dim=1) # b x 2 x 512
        channels = [None] * self.inject_depth + [latent]
        encoded_audio = self.autoencoder.encode(audio)
        
        # Adapt input to diffusion if adapter provided
        # x = self.adapter.encode(x) if exists(self.adapter) else x
        # Compute diffusion loss
        loss = self.diffusion_model(encoded_audio, 
                                    text=text, 
                                    channels=channels, 
                                    embedding_mask_proba=0.1,
                                    **kwargs)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.diffusion_model.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)
    
    def train_dataloader(self):
        train_loader = DataLoader(self.dataset, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, batch_size=self.batch_size, shuffle=True)
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


    # Hyperparameters for the model
    parser.add_argument("--dataset_path", type=str, default="/home/gconcialdi/spotdl/")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--crop", type=int, default=2**20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)



    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()
    
    logger = CometLogger(
        api_key="9LmOAqSG4omncUN3QT42iQoqb",
        project_name="ainur",
        workspace="gio99c",
        experiment_name="ainur",
        offline=True
        )


    diffusion = Ainur(inject_depth=0, dataset_path=args.dataset_path, num_workers=1, batch_size=args.batch_size)

    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      precision=args.precision,
                      accelerator=args.accelerator,
                      devices=args.n_devices,
                      num_nodes=args.num_nodes,
                      default_root_dir=args.default_root_dir,
                      plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
                      callbacks=[StochasticWeightAveraging(swa_lrs=1e-4)])

    trainer.fit(diffusion, ckpt_path=args.checkpoint_path)
