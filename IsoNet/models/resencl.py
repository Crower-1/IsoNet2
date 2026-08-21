"""Dense ResEnc-L encoder with an IsoNet2 restoration decoder.

The encoder mirrors nnSSL's ``ResEncL`` (dynamic-network-architectures'
``ResidualEncoder`` with ``BasicBlockD``), but is implemented with plain
PyTorch modules. SparK masks, mask tokens, and the MAE decoder are deliberately
not part of this model.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


RESENCL_CHANNELS = (32, 64, 128, 256, 320, 320)
RESENCL_BLOCKS = (1, 3, 4, 6, 6, 6)
RESENCL_STRIDES = (1, 2, 2, 2, 2, 2)


class ConvNormAct3D(nn.Module):
    """Conv3d -> InstanceNorm3d -> optional LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        bias: bool = True,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin(self.norm(self.conv(x)))


class _Stem3D(nn.Module):
    """Keep nnSSL's canonical ``stem.convs.0`` state-dict namespace."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(ConvNormAct3D(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class ResidualBlock3D(nn.Module):
    """ResNet-D basic block used by nnSSL ResEncL.

    Downsampling uses AvgPool3d on the skip branch, followed by a bias-free
    1x1 projection only when the channel count changes.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels, stride=stride)
        self.conv2 = ConvNormAct3D(out_channels, out_channels, activation=False)
        self.nonlin2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        skip: list[nn.Module] = []
        if stride != 1:
            skip.append(nn.AvgPool3d(kernel_size=stride, stride=stride))
        if in_channels != out_channels:
            skip.append(
                ConvNormAct3D(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    activation=False,
                )
            )
        self.skip = nn.Sequential(*skip) if skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nonlin2(self.conv2(self.conv1(x)) + self.skip(x))


class _ResidualStage3D(nn.Module):
    """Keep nnSSL's canonical ``stages.N.blocks.M`` namespace."""

    def __init__(self, in_channels: int, out_channels: int, blocks: int, stride: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock3D(in_channels, out_channels, stride=stride),
            *[ResidualBlock3D(out_channels, out_channels) for _ in range(1, blocks)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResEncLEncoder3D(nn.Module):
    """Dense six-stage nnSSL ResEncL encoder returning all skip features."""

    channels = RESENCL_CHANNELS
    blocks_per_stage = RESENCL_BLOCKS
    strides = RESENCL_STRIDES

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = _Stem3D(in_channels, self.channels[0])
        stages = []
        stage_input = self.channels[0]
        for output, blocks, stride in zip(self.channels, self.blocks_per_stage, self.strides):
            stages.append(_ResidualStage3D(stage_input, output, blocks, stride))
            stage_input = output
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class ResEncLDecoder3D(nn.Module):
    """nnU-Net-style decoder consuming ResEncL skip features.

    Each level uses a stride-2 ConvTranspose3d, concatenates the corresponding
    encoder skip, and applies one Conv3d/InstanceNorm/LeakyReLU fusion block.
    This matches the decoder depth used by nnSSL's ResEncL plan while keeping
    the decoder randomly initialized for restoration.
    """

    def __init__(self, channels: Sequence[int] = RESENCL_CHANNELS) -> None:
        super().__init__()
        decoder_channels = tuple(reversed(channels[:-1]))  # 320, 256, 128, 64, 32
        transpconvs = []
        stages = []
        current = channels[-1]
        for skip_channels, output in zip(reversed(channels[:-1]), decoder_channels):
            transpconvs.append(
                nn.ConvTranspose3d(
                    current, output, kernel_size=2, stride=2, bias=True
                )
            )
            stages.append(ConvNormAct3D(output + skip_channels, output))
            current = output
        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages = nn.ModuleList(stages)
        self.output_channels = current

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != 6:
            raise ValueError(f"ResEncL decoder expects 6 feature maps, got {len(features)}")
        x = features[-1]
        for transpconv, stage, skip in zip(
            self.transpconvs, self.stages, reversed(features[:-1])
        ):
            x = transpconv(x)
            if x.shape[-3:] != skip.shape[-3:]:
                raise RuntimeError(
                    f"Decoder/skip shape mismatch: {tuple(x.shape)} vs {tuple(skip.shape)}"
                )
            x = stage(torch.cat((x, skip), dim=1))
        return x


class ResEncLRestorationNet(nn.Module):
    """nnSSL-compatible encoder plus residual one-channel restoration head."""

    required_divisor = 32

    def __init__(
        self, in_channels: int = 1, out_channels: int = 1, encoder_input_sign: float = -1.0
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("Residual restoration requires matching input/output channels")
        self.encoder = ResEncLEncoder3D(in_channels)
        self.decoder = ResEncLDecoder3D()
        self.output_head = nn.Conv3d(self.decoder.output_channels, out_channels, kernel_size=1, bias=True)
        if float(encoder_input_sign) not in {-1.0, 1.0}:
            raise ValueError("encoder_input_sign must be either -1 or +1")
        self.register_buffer(
            "encoder_input_sign", torch.tensor(float(encoder_input_sign)), persistent=True
        )
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B, C, D, H, W], got {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected one input channel, got {x.shape[1]}")
        invalid = [size for size in x.shape[-3:] if size % ResEncLRestorationNet.required_divisor]
        if invalid:
            raise ValueError(
                f"Every spatial dimension must be divisible by 32, got {tuple(x.shape[-3:])}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        residual = self.output_head(self.decoder(self.encoder(x * self.encoder_input_sign)))
        return x + residual

    def set_encoder_trainable(self, mode: str) -> None:
        """Configure ``frozen``, ``deep`` (stages 3-5), or ``all`` fine-tuning."""
        mode = str(mode).lower()
        if mode not in {"frozen", "deep", "all"}:
            raise ValueError("encoder_trainable must be one of: frozen, deep, all")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = mode == "all"
        if mode == "deep":
            for stage in self.encoder.stages[3:]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.encoder.parameters()

    def restoration_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.decoder.parameters()
        yield from self.output_head.parameters()
