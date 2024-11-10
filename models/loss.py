from typing import List
import torch
from torch import nn

from models.modules import ResNet2d


class WaveDiscriminator(nn.Module):
    r"""MelGAN discriminator from https://arxiv.org/pdf/1910.06711.pdf"""

    def __init__(self, resolution: int = 1, n_channels: int = 4) -> None:
        super().__init__()
        assert resolution >= 1
        if resolution == 1:
            self.avg_pool = nn.Identity()
        else:
            self.avg_pool = nn.AvgPool1d(resolution * 2, stride=resolution)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.layers = nn.ModuleList(
            [
                nn.utils.weight_norm(
                    nn.Conv1d(1, n_channels, kernel_size=15, padding=7)
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        n_channels,
                        4 * n_channels,
                        kernel_size=41,
                        stride=4,
                        padding=20,
                        groups=4,
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        4 * n_channels,
                        16 * n_channels,
                        kernel_size=41,
                        stride=4,
                        padding=20,
                        groups=16,
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        16 * n_channels,
                        64 * n_channels,
                        kernel_size=41,
                        stride=4,
                        padding=20,
                        groups=64,
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        64 * n_channels,
                        256 * n_channels,
                        kernel_size=41,
                        stride=4,
                        padding=20,
                        groups=256,
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(
                        256 * n_channels, 256 * n_channels, kernel_size=5, padding=2
                    )
                ),
                nn.utils.weight_norm(
                    nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.avg_pool(x)
        feats = []
        for layer in self.layers[:-1]:
            x = layer(x)
            feats.append(x)
            x = self.activation(x)
        feats.append(self.layers[-1](x))
        return feats


class STFTDiscriminator(nn.Module):
    r"""STFT-based discriminator from https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(
        self, n_fft: int = 1024, hop_length: int = 256, n_channels: int = 32
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding="same"),
            nn.LeakyReLU(0.3, inplace=True),
            ResNet2d(n_channels, 2, stride=(2, 1)),
            ResNet2d(2 * n_channels, 2, stride=(2, 2)),
            ResNet2d(4 * n_channels, 1, stride=(2, 1)),
            ResNet2d(4 * n_channels, 2, stride=(2, 2)),
            ResNet2d(8 * n_channels, 1, stride=(2, 1)),
            ResNet2d(8 * n_channels, 2, stride=(2, 2)),
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1)),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] == 1
        # input: [batch, channel, sequence]
        x = torch.squeeze(input, 1).to(
            torch.float32
        )  # torch.stft() doesn't accept float16
        x = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            normalized=True,
            onesided=True,
            return_complex=True,
        )
        x = torch.abs(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.layers(x)
        return x


class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2**i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(
                input,
                n_fft=s,
                hop_length=s // 4,
                win_length=s,
                normalized=True,
                onesided=True,
                return_complex=True,
            )
            x = torch.abs(x)
            y = torch.stft(
                target,
                n_fft=s,
                hop_length=s // 4,
                win_length=s,
                normalized=True,
                onesided=True,
                return_complex=True,
            )
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, : y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, : x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(
                torch.square(torch.log(x + self.eps) - torch.log(y + self.eps))
            )
        return loss / (12 - 6)


class ReconstructionLoss2(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf"""

    def __init__(self, sample_rate, eps=1e-5):
        super().__init__()
        import torchaudio

        self.layers = nn.ModuleList()
        self.alpha = []
        self.eps = eps
        for i in range(6, 12):
            melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(2**i),
                win_length=int(2**i),
                hop_length=int(2**i / 4),
                n_mels=64,
            )
            self.layers.append(melspec)
            self.alpha.append((2**i / 2) ** 0.5)

    def forward(self, input, target):
        loss = 0
        for alpha, melspec in zip(self.alpha, self.layers):
            x = melspec(input)
            y = melspec(target)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(
                torch.square(torch.log(x + self.eps) - torch.log(y + self.eps))
            )
        return loss
