#!/usr/bin/env python


import time

import numpy as np
import torch
import torch.nn as nn
from model import ARCPositionalEncoding as OldARCPositionalEncoding


class ARCPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, grid_dim: int, num_train_pairs: int):
        super().__init__()
        self.d_model = d_model
        self.grid_dim = grid_dim
        self.num_train_pairs = num_train_pairs

        # Embeddings for row and column positions
        self.row_embedding = nn.Embedding(self.grid_dim, self.d_model // 4)
        self.col_embedding = nn.Embedding(self.grid_dim, self.d_model // 4)

        # Embedding for input vs output
        self.input_output_embedding = nn.Embedding(2, d_model // 4)

        # Embedding for training pair index
        self.pair_embedding = nn.Embedding(
            self.num_train_pairs + 1, d_model // 4
        )  # +1 for test pair

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, num_grids, height, width, _ = x.size()
        device = x.device

        # Row pos embedding
        row_pos = torch.arange(height, device=device)
        row_emb = (
            self.row_embedding.forward(row_pos)
            .unsqueeze(1)
            .expand(num_grids, -1, width, -1)
        )

        # Column pos embedding
        col_pos = torch.arange(width, device=device)
        col_emb = (
            self.col_embedding.forward(col_pos)
            .unsqueeze(0)
            .expand(num_grids, height, -1, -1)
        )

        # Input/output embedding
        grid_indices = torch.arange(num_grids, device=device)
        is_output = (grid_indices % 2 == 1).long()
        io_emb = (
            self.input_output_embedding(is_output)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(num_grids, height, width, -1)
        )

        # Pair embedding
        pair_indices = torch.div(grid_indices, 2, rounding_mode="floor")
        pair_indices[-1] = self.num_train_pairs
        pair_emb = (
            self.pair_embedding(pair_indices)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(num_grids, height, width, -1)
        )

        # Combine all embeddings (1, num_grids, height, width, d_model)
        combined_emb = torch.cat([row_emb, col_emb, io_emb, pair_emb], dim=-1)

        return combined_emb


def test_arc_positional_encoding():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    d_model = 512
    grid_dim = 30
    num_train_pairs = 5
    batch_size = 4
    num_grids = 11
    height = 25
    width = 25

    # Initialize both versions of the encoding
    old_encoding = OldARCPositionalEncoding(d_model, grid_dim, num_train_pairs)
    new_encoding = ARCPositionalEncoding(d_model, grid_dim, num_train_pairs)

    # Generate random input data
    x = torch.randn(batch_size, num_grids, height, width, d_model)

    # Compute outputs
    old_time = time.time()
    for i in range(1):
        old_output = new_encoding.forward_old(x)
    print("old", old_output.shape, time.time() - old_time)

    new_time = time.time()
    for i in range(1):
        new_output = new_encoding.forward(x)
    print("new", new_output.shape, time.time() - new_time)

    # Compare outputs
    is_close = torch.allclose(old_output, new_output, rtol=1e-5, atol=1e-5)

    if is_close:
        print("Test passed: Both versions produce the same output.")
    else:
        print("Test failed: Outputs are different.")
        max_diff = torch.max(torch.abs(old_output - new_output))
        print(f"Maximum difference: {max_diff}")


if __name__ == "__main__":
    test_arc_positional_encoding()
