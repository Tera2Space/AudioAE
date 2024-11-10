from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from itertools import chain
import random
import torchaudio
from torch import Tensor
from models.audioae import AudioAE
from models.loss import WaveDiscriminator, ReconstructionLoss, STFTDiscriminator


class DatasetMy:
    def __init__(self, path) -> None:
        # reads file with audio pathes separated by \n, prefered to use wav
        with open(path, "r") as file_d:
            self.samples = file_d.read().split("\n")
        self.samples = [i for i in self.samples if i]

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        path = self.samples[n]
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

    def __len__(self) -> int:
        return len(self.samples)


class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_rate, segment_length):
        self._dataset = dataset
        self._sample_rate = sample_rate
        self._segment_length = segment_length

    def __getitem__(self, index):
        x, sample_rate = self._dataset[index]
        assert sample_rate == self._sample_rate
        assert x.shape[0] == 1
        x = torch.squeeze(x)
        x *= 0.95 / torch.max(x)
        assert x.dim() == 1
        if x.shape[0] < self._segment_length:
            x = F.pad(x, [0, self._segment_length - x.shape[0]], "constant")
        pos = random.randint(0, x.shape[0] - self._segment_length)
        x = x[pos : pos + self._segment_length]
        return x

    def __len__(self):
        return len(self._dataset)


class StreamableModelPL(pl.LightningModule, AudioAE):
    def __init__(
        self,
        n_channels: int = 32,
        padding: str = "valid",
        batch_size: int = 32,
        sample_rate: int = 24000,
        segment_length: int = 32270,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        dataset_path="./filelist.train",
    ) -> None:
        pl.LightningModule.__init__(self)
        AudioAE.__init__(self, n_channels, padding)

        hparams = {
            "n_channels": n_channels,
            "padding": padding,
            "batch_size": batch_size,
            "sample_rate": sample_rate,
            "segment_length": segment_length,
            "lr": lr,
            "b1": b1,
            "b2": b2,
            "dataset_path": dataset_path,
        }
        self.save_hyperparameters(hparams)
        self.automatic_optimization = False

        self.wave_discriminators = nn.ModuleList(
            [
                WaveDiscriminator(resolution=1),
                WaveDiscriminator(resolution=2),
                WaveDiscriminator(resolution=4),
            ]
        )
        self.rec_loss = ReconstructionLoss()
        self.stft_discriminator = STFTDiscriminator(n_channels=n_channels)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        optimizer_g = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=lr,
            betas=(b1, b2),
        )
        optimizer_d = torch.optim.Adam(
            chain(
                self.wave_discriminators.parameters(),
                self.stft_discriminator.parameters(),
            ),
            lr=lr,
            betas=(b1, b2),
        )
        return [optimizer_g, optimizer_d], []

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        input = batch[:, None, :]
        # input: [batch, channel, sequence]

        # train generator
        self.toggle_optimizer(optimizer_g)
        output = self.forward(input)

        stft_out = self.stft_discriminator(output)
        g_stft_loss = torch.mean(torch.relu(1 - stft_out))
        self.log("g_stft_loss", g_stft_loss)

        g_wave_loss = 0
        g_feat_loss = 0
        for i in range(3):
            feats1 = self.wave_discriminators[i](input)
            feats2 = self.wave_discriminators[i](output)
            assert len(feats1) == len(feats2)
            g_wave_loss += torch.mean(torch.relu(1 - feats2[-1]))
            g_feat_loss += sum(
                torch.mean(torch.abs(f1 - f2))
                for f1, f2 in zip(feats1[:-1], feats2[:-1])
            ) / (len(feats1) - 1)
        self.log("g_wave_loss", g_wave_loss / 3)
        self.log("g_feat_loss", g_feat_loss / 3)

        g_rec_loss = self.rec_loss(output[:, 0, :], input[:, 0, :])
        self.log("g_rec_loss", g_rec_loss, prog_bar=True)

        g_feat_loss = g_feat_loss / 3
        g_adv_loss = (g_stft_loss + g_wave_loss) / 4
        g_loss = g_adv_loss + 100 * g_feat_loss + g_rec_loss
        self.log("g_loss", g_loss, prog_bar=True)

        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        self.toggle_optimizer(optimizer_d)
        output = self.forward(input)

        stft_out = self.stft_discriminator(input)
        d_stft_loss = torch.mean(torch.relu(1 - stft_out))
        stft_out = self.stft_discriminator(output)
        d_stft_loss += torch.mean(torch.relu(1 + stft_out))

        d_wave_loss = 0
        for i in range(3):
            feats = self.wave_discriminators[i](input)
            d_wave_loss += torch.mean(torch.relu(1 - feats[-1]))
            feats = self.wave_discriminators[i](output)
            d_wave_loss += torch.mean(torch.relu(1 + feats[-1]))

        d_loss = (d_stft_loss + d_wave_loss) / 4

        self.log("d_stft_loss", d_stft_loss)
        self.log("d_wave_loss", d_wave_loss / 3)

        d_loss = (d_stft_loss + d_wave_loss) / 4
        self.log("d_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def train_dataloader(self):
        return self._make_dataloader(True)

    def _make_dataloader(self, train: bool):
        def collate(examples):
            return torch.stack(examples)

        ds = DatasetMy(self.hparams.dataset_path)

        ds = VoiceDataset(ds, self.hparams.sample_rate, self.hparams.segment_length)

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            collate_fn=collate,
            num_workers=8,
        )
        return loader


def train():
    model = StreamableModelPL(
        n_channels=32,
        batch_size=64,
        sample_rate=24000,
        segment_length=32270,
        padding="same",
        dataset_path="./filelist.train",
    )
    trainer = pl.Trainer(
        max_epochs=10000,
        log_every_n_steps=2,
        precision="16-mixed",
        logger=pl.loggers.TensorBoardLogger("lightning_logs", name="soundstream"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_last=True, every_n_train_steps=3125),
        ],
    )
    trainer.fit(
        model,
        # ckpt_path="last.ckpt",
    )

    return model


if __name__ == "__main__":
    train()
