"""Architectures used in the BaCNet model-ablation study.

All models consume the same 7,808-dimensional concatenated input in this
order: ESM-2, Morgan fingerprint, Chemical Checker, and ChemBERTa.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


MODALITY_DIMS = OrderedDict(
    protein=5120,
    morgan=1024,
    chemical_checker=1280,
    chemberta=384,
)


class BaselineMLP(nn.Module):
    """Original BaCNet MLP used as the ablation baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (1024, 256, 32),
        dropout_rate: float = 0.01,
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 3:
            raise ValueError("hidden_dims must contain exactly three dimensions")
        blocks = []
        current_dim = input_dim
        for output_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Linear(current_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                )
            )
            current_dim = output_dim
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(current_dim, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for index, block in enumerate(self.blocks):
            output = block(output)
            if index < len(self.blocks) - 1:
                output = self.dropout(output)
        return self.output(output)


class ResidualLinearBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.shortcut = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.dropout(self.activation(self.main(inputs) + self.shortcut(inputs)))


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (1024, 256, 32),
        dropout_rate: float = 0.01,
    ) -> None:
        super().__init__()
        dimensions = [input_dim, *hidden_dims]
        self.blocks = nn.ModuleList(
            ResidualLinearBlock(input_dim_, output_dim, dropout_rate)
            for input_dim_, output_dim in zip(dimensions[:-1], dimensions[1:])
        )
        self.output = nn.Linear(dimensions[-1], 1)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for block in self.blocks:
            output = block(output)
        return self.output(output)


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class ProjectedFusionModel(nn.Module):
    """Project individual modalities before concat, residual, or gated fusion."""

    VALID_FUSIONS = {"concat", "residual", "gated"}

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        projection_dim: int = 256,
        hidden_dims: Sequence[int] = (1024, 256, 32),
        dropout_rate: float = 0.01,
        fusion: str = "concat",
        residual_mlp: bool = False,
    ) -> None:
        super().__init__()
        if fusion not in self.VALID_FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(self.VALID_FUSIONS)}")
        if not modality_dims:
            raise ValueError("At least one modality is required")
        if fusion == "residual" and len(modality_dims) < 2:
            raise ValueError("Residual fusion requires at least two modalities")

        self.modality_names = tuple(modality_dims.keys())
        self.modality_dims = tuple(modality_dims.values())
        self.fusion = fusion
        self.projectors = nn.ModuleDict(
            {
                name: ModalityProjector(dimension, projection_dim, dropout_rate)
                for name, dimension in modality_dims.items()
            }
        )
        if fusion == "concat":
            fused_dim = projection_dim * len(modality_dims)
            self.fusion_norm = nn.Identity()
            self.gate = None
        else:
            fused_dim = projection_dim
            self.fusion_norm = nn.LayerNorm(projection_dim)
            self.gate = (
                nn.Linear(projection_dim * len(modality_dims), len(modality_dims))
                if fusion == "gated"
                else None
            )
        head_class = ResidualMLP if residual_mlp else BaselineMLP
        self.head = head_class(fused_dim, hidden_dims, dropout_rate)

    def _project(self, inputs: Tensor) -> list[Tensor]:
        expected_dim = sum(self.modality_dims)
        if inputs.ndim != 2 or inputs.shape[1] != expected_dim:
            raise ValueError(
                f"Expected input shape (batch, {expected_dim}), got {tuple(inputs.shape)}"
            )
        chunks = torch.split(inputs, self.modality_dims, dim=1)
        return [
            self.projectors[name](chunk)
            for name, chunk in zip(self.modality_names, chunks)
        ]

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self._project(inputs)
        if self.fusion == "concat":
            fused = torch.cat(projected, dim=1)
        elif self.fusion == "residual":
            chemical_residual = torch.stack(projected[1:], dim=1).mean(dim=1)
            fused = self.fusion_norm(projected[0] + chemical_residual)
        else:
            concatenated = torch.cat(projected, dim=1)
            weights = torch.softmax(self.gate(concatenated), dim=1)
            stacked = torch.stack(projected, dim=1)
            fused = self.fusion_norm((stacked * weights.unsqueeze(-1)).sum(dim=1))
        return self.head(fused)


ABLATION_VARIANTS = (
    "baseline",
    "residual_mlp",
    "projected_concat",
    "projected_residual_fusion",
    "projected_gated_fusion",
    "projected_concat_residual_mlp",
)

ABLATION_DESCRIPTIONS = {
    "baseline": "Raw 7808-d concat followed by the original MLP",
    "residual_mlp": "Raw concat followed by an MLP with projected shortcuts",
    "projected_concat": "Equal-dimensional per-modality projections, then concat",
    "projected_residual_fusion": "Protein projection plus mean chemical residual",
    "projected_gated_fusion": "Softmax-gated sum of projected modalities",
    "projected_concat_residual_mlp": (
        "Projected concat followed by an MLP with projected shortcuts"
    ),
}


def create_ablation_model(
    variant: str,
    modality_dims: Mapping[str, int] = MODALITY_DIMS,
    projection_dim: int = 256,
    hidden_dims: Sequence[int] = (1024, 256, 32),
    dropout_rate: float = 0.01,
) -> nn.Module:
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {ABLATION_VARIANTS}")
    input_dim = sum(modality_dims.values())
    if variant == "baseline":
        return BaselineMLP(input_dim, hidden_dims, dropout_rate)
    if variant == "residual_mlp":
        return ResidualMLP(input_dim, hidden_dims, dropout_rate)

    fusion = {
        "projected_concat": "concat",
        "projected_residual_fusion": "residual",
        "projected_gated_fusion": "gated",
        "projected_concat_residual_mlp": "concat",
    }[variant]
    return ProjectedFusionModel(
        modality_dims=modality_dims,
        projection_dim=projection_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        fusion=fusion,
        residual_mlp=variant == "projected_concat_residual_mlp",
    )
