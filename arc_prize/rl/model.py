from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc_prize.model import (
    ARCVisionEncoder,
    DecoderLayerWithAttention,
    DecoderWithAttention,
)


@dataclass(frozen=True)
class ValueNetworkParams:
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    dropout: float


class ValueNetwork(nn.Module):
    params: ValueNetworkParams

    def __init__(self, params: ValueNetworkParams):
        super().__init__()
        self.params = params

        self.value_token = nn.Parameter(torch.randn(1, 1, self.params.d_model))

        decoder_layer = DecoderLayerWithAttention(
            d_model=self.params.d_model,
            nhead=self.params.num_heads,
            dim_feedforward=self.params.d_ff,
            dropout=self.params.dropout,
        )
        self.decoder = DecoderWithAttention(
            decoder_layer, num_layers=self.params.num_layers
        )

        self.value_head = nn.Linear(self.params.d_model, 1)

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)

        value_tgt = self.value_token.expand(batch_size, -1, -1)

        value_output = self.decoder.forward(
            tgt=value_tgt, memory=input, memory_key_padding_mask=mask
        )

        logits = torch.sigmoid(self.value_head(value_output))
        return logits


class TrajectorySearch:
    def __init__(
        self,
        model: ARCVisionEncoder,
        policy_network: TrajectoryPolicy,
        temperatures: list[float],
        beam_width: int = 3,
        max_depth: int = 5,
        stop_threshold: float = 0.9,
        ensemble_k: int = 2,
    ):
        self.model = model
        self.policy_network = policy_network
        self.temperatures = temperatures
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.stop_threshold = stop_threshold
        self.ensemble_k = ensemble_k

    def ensemble_select(
        self, finished_trajectories: List[TrajectoryState]
    ) -> Tuple[torch.Tensor, List[TrajectoryState]]:
        """
        Select final output by ensembling top-k trajectories

        Returns:
            ensemble_grid: Grid produced by majority voting
            used_trajectories: List of trajectories used in ensemble
        """
        if not finished_trajectories:
            raise ValueError("No finished trajectories to select from")

        # Get top-k trajectories by value
        top_k = sorted(finished_trajectories, key=lambda x: x.value, reverse=True)[
            : self.ensemble_k
        ]

        # Stack grids and get their values for weighted voting
        grids = torch.stack([t.grid for t in top_k])
        values = torch.tensor([t.value for t in top_k])
        weights = F.softmax(values, dim=0)

        # For each position, do weighted voting across all classes
        batch, height, width = grids.shape
        ensemble_grid = torch.zeros_like(grids[0])

        for h in range(height):
            for w in range(width):
                # Get votes for this position
                votes = grids[:, h, w]  # [k]

                # Count weighted votes for each class
                vote_counts = torch.zeros(
                    self.vision_model.num_classes, device=grids.device
                )
                for vote, weight in zip(votes, weights):
                    vote_counts[vote] += weight

                # Select class with highest weighted votes
                ensemble_grid[h, w] = vote_counts.argmax()

        return ensemble_grid, top_k

    @torch.no_grad()
    def search(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[float], dict]:
        """
        Returns:
            best_grid: Ensembled grid from top-k trajectories
            temp_history: Temperature sequences used (from highest-value trajectory)
            stats: Search statistics including ensemble information
        """
        # Run search to get finished trajectories
        finished_trajectories = self.run_search(src, src_mask)

        # Ensemble select best output
        ensemble_grid, used_trajectories = self.ensemble_select(finished_trajectories)

        # Get temperature history from highest-value trajectory
        best_trajectory = max(used_trajectories, key=lambda x: x.value)
        trajectory_history = best_trajectory.get_trajectory()
        temp_history = [step[1] for step in trajectory_history[1:]]

        # Collect stats about the ensemble
        stats = {
            "num_total_trajectories": len(finished_trajectories),
            "ensemble_size": len(used_trajectories),
            "ensemble_values": [t.value for t in used_trajectories],
            "ensemble_steps": [t.step for t in used_trajectories],
            "ensemble_agreement": self._compute_agreement(used_trajectories),
            "best_value": best_trajectory.value,
            "best_steps": best_trajectory.step,
        }

        return ensemble_grid, temp_history, stats

    def _compute_agreement(self, trajectories: List[TrajectoryState]) -> float:
        """
        Compute average agreement between trajectories in ensemble
        """
        grids = [t.grid for t in trajectories]
        agreement = 0
        count = 0

        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                agreement += (grids[i] == grids[j]).float().mean().item()
                count += 1

        return agreement / count if count > 0 else 1.0

    def visualize_ensemble(
        self, trajectories: List[TrajectoryState], ensemble_grid: torch.Tensor
    ):
        """
        Visualize the ensemble process and agreement between trajectories
        """
        # This would be implemented based on your visualization preferences
        # Could show:
        # - Individual grids from each trajectory
        # - Heatmap of agreement across positions
        # - Final ensemble result
        # - Trajectory values and weights used
        pass
