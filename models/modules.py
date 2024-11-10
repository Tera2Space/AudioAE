from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

# ! Help me with ConvNeXtBlock plz

# class AdaLayerNorm(nn.Module):
#     """
#     Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

#     Args:
#         num_embeddings (int): Number of embeddings.
#         embedding_dim (int): Dimension of the embeddings.
#     """

#     def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps
#         self.dim = embedding_dim
#         self.scale = nn.Embedding(
#             num_embeddings=num_embeddings, embedding_dim=embedding_dim
#         )
#         self.shift = nn.Embedding(
#             num_embeddings=num_embeddings, embedding_dim=embedding_dim
#         )
#         torch.nn.init.ones_(self.scale.weight)
#         torch.nn.init.zeros_(self.shift.weight)

#     def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
#         scale = self.scale(cond_embedding_id)
#         shift = self.shift(cond_embedding_id)
#         x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
#         x = x * scale + shift
#         return x


# class ConvNeXtBlock(nn.Module):
#     def __init__(
#         self,
#         n_channels,
#         intermediate_dim,
#         kernel_size: int = 7,
#         padding: str = "valid",
#         dilation: int = 1,
#         layer_scale_init_value: float = 0.0,
#         adanorm_num_embeddings=None,
#     ):
#         super().__init__()
#         assert padding in ["valid", "same"]
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.dilation = dilation
#         self._padding_size = (kernel_size // 2) * dilation

#         self.dwconv = nn.Conv1d(
#             n_channels,
#             n_channels,
#             kernel_size=kernel_size,
#             padding=padding,
#             groups=n_channels,
#             dilation=dilation,
#         )  # depthwise conv
#         self.adanorm = adanorm_num_embeddings is not None
#         if adanorm_num_embeddings:
#             self.norm = AdaLayerNorm(adanorm_num_embeddings, n_channels, eps=1e-6)
#         else:
#             self.norm = nn.LayerNorm(n_channels, eps=1e-6)
#         self.pwconv1 = nn.Linear(
#             n_channels, intermediate_dim
#         )  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(intermediate_dim, n_channels)
#         self.gamma = (
#             nn.Parameter(
#                 layer_scale_init_value * torch.ones(n_channels), requires_grad=True
#             )
#             if layer_scale_init_value > 0
#             else None
#         )

#     def forward(self, x: torch.Tensor, cond_embedding_id=None) -> torch.Tensor:
#         residual = x
#         x = self.dwconv(x)
#         x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
#         if self.adanorm:
#             assert cond_embedding_id is not None
#             x = self.norm(x, cond_embedding_id)
#         else:
#             x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

#         if self.padding == "valid":
#             residual = residual[:, :, self._padding_size : -self._padding_size]
#         x = residual + x
#         return x


class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = "valid",
        dilation: int = 1,
    ) -> None:
        super().__init__()
        assert padding in ["valid", "same"]
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv1 = nn.Conv1d(n_channels, n_channels, kernel_size=1)

    def forward(self, input):
        y = input
        x = self.conv0(input)
        x = F.elu(x)
        x = self.conv1(x)
        if self.padding == "valid":
            y = y[:, :, self._padding_size : -self._padding_size]
        x += y
        x = F.elu(x)
        return x


class ResNet2d(nn.Module):
    def __init__(self, n_channels: int, factor: int, stride: Tuple[int, int]) -> None:
        # https://arxiv.org/pdf/2005.00341.pdf
        # The original paper uses layer normalization, but here
        # we use batch normalization.
        super().__init__()
        self.conv0 = nn.Conv2d(
            n_channels, n_channels, kernel_size=(3, 3), padding="same"
        )
        self.bn0 = nn.BatchNorm2d(n_channels)
        self.conv1 = nn.Conv2d(
            n_channels,
            factor * n_channels,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(factor * n_channels)
        self.conv2 = nn.Conv2d(
            n_channels, factor * n_channels, kernel_size=1, stride=stride
        )
        self.bn2 = nn.BatchNorm2d(factor * n_channels)
        self.pad = nn.ReflectionPad2d(
            [
                (stride[1] + 1) // 2,
                (stride[1] + 2) // 2,
                (stride[0] + 1) // 2,
                (stride[0] + 2) // 2,
            ]
        )
        self.activation = nn.LeakyReLU(0.3)

    def forward(self, input):
        x = self.conv0(input)
        x = self.bn0(x)
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.bn1(x)

        # shortcut
        y = self.conv2(input)
        y = self.bn2(y)

        x += y
        x = self.activation(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_channels: int, padding: str, stride: int) -> None:
        super().__init__()
        assert padding in ["valid", "same"]
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            # ConvNeXtBlock(
            #     n_channels // 2,
            #     n_channels // 2,
            #     padding=padding,
            #     dilation=1,
            #     layer_scale_init_value=1 / 3,
            # ),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            # ConvNeXtBlock(
            #     n_channels // 2,
            #     n_channels // 2,
            #     padding=padding,
            #     dilation=3,
            #     layer_scale_init_value=1 / 3,
            # ),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            # ConvNeXtBlock(
            #     n_channels // 2,
            #     n_channels // 2,
            #     padding=padding,
            #     dilation=9,
            #     layer_scale_init_value=1 / 3,
            # ),
            nn.Conv1d(
                n_channels // 2,
                n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == "same" else 0,
                stride=stride,
            ),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class DecoderBlock(nn.Module):
    def __init__(self, n_channels: int, padding: str, stride: int) -> None:
        super().__init__()
        assert padding in ["valid", "same"]
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels,
                n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == "same" else 0,
                stride=stride,
            ),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            # ConvNeXtBlock(
            #     n_channels // 2,
            #     n_channels // 2,
            #     padding=padding,
            #     dilation=1,
            #     layer_scale_init_value=1 / 3,
            # ),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            # ConvNeXtBlock(
            #     n_channels // 2,
            #     n_channels // 2,
            #     padding=padding,
            #     dilation=3,
            #     layer_scale_init_value=1 / 3,
            # ),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            # ConvNeXtBlock(
            #     n_channels // 2,
            #     n_channels // 2,
            #     padding=padding,
            #     dilation=9,
            #     layer_scale_init_value=1 / 3,
            # ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
