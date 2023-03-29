import comet_ml

import os
import signal
import lightning as L
import torch
import torchaudio
from archisound import ArchiSound
from data.dataset import get_dataset
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers.comet import CometLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader


class LitDAE(L.LightningModule):
    def __init__(self, dataset_path, crop=2**18, batch_size=64, num_workers=16):
        super(LitDAE, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = get_dataset(dataset_path, crop=crop)
        self.autoencoder = ArchiSound.from_pretrained("dmae1d-ATC32-v3")

    def training_step(self, batch, batch_idx):
        audio, *_ = batch
        batch_size = audio.shape[0]
        loss = self.autoencoder(audio)
        self.log('loss_ae', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)

        # Log audio 
        if batch_idx == 0:
            tmp_dir = os.path.join(os.getcwd(), ".tmp")
            if not os.path.exists(tmp_dir):
                os.mkdir(tmp_dir)
            torchaudio.save(os.path.join(tmp_dir, "original.wav"), audio[0], 48_000) 
            torchaudio.save(os.path.join(tmp_dir, "reconstructed.wav"), self.autoencoder.decode(self.autoencoder.encode(audio[:1,:,:]), num_steps=10).squeeze(), 48_000)
            self.logger.experiment.log_audio(os.path.join(tmp_dir, "original.wav"))
            self.logger.experiment.log_audio(os.path.join(tmp_dir, "reconstructed.wav"))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.autoencoder.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-6, weight_decay=1e-3)
    
    def train_dataloader(self): ##workers
        train_loader = DataLoader(self.dataset, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, batch_size=self.batch_size, shuffle=True)
        return train_loader
    
    def encode(self, x):
        return self.autoencoder.encode(x)
    
    def decode(self, x, num_steps=10):
        return self.autoencoder.decode(x, num_steps=num_steps)




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
        experiment_name="dae",
        offline=False
        )


    autoencoder = LitDAE(crop=args.crop, batch_size=args.batch_size, dataset_path=args.dataset_path, num_workers=args.num_workers)

    trainer = Trainer(max_epochs=args.epochs,
                      logger=logger,
                      precision=args.precision,
                      accelerator=args.accelerator,
                      devices=args.n_devices,
                      num_nodes=args.num_nodes,
                      default_root_dir=args.default_root_dir,
                      plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
                      callbacks=[StochasticWeightAveraging(swa_lrs=1e-4)])

    trainer.fit(autoencoder, ckpt_path=args.checkpoint_path)