import torch
from torch import nn
from models.modules import EncoderBlock, DecoderBlock


class Encoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ["valid", "same"]
        self.layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            EncoderBlock(2 * n_channels, padding=padding, stride=2),
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            EncoderBlock(8 * n_channels, padding=padding, stride=5),
            EncoderBlock(16 * n_channels, padding=padding, stride=8),
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=3, padding=padding),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Decoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ["valid", "same"]
        self.layers = nn.Sequential(
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            DecoderBlock(16 * n_channels, padding=padding, stride=8),
            DecoderBlock(8 * n_channels, padding=padding, stride=5),
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class AudioAE(nn.Module):
    def __init__(
        self,
        n_channels: int = 32,
        padding: str = "valid",
    ) -> None:
        super().__init__()
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        return x
