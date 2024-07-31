from typing import Optional

import torch
import torch.nn as nn


class ARCTransformer(nn.Module):
    grid_dim: int
    num_train_pairs: int
    num_classes: int
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    dropout: float

    def __init__(
        self,
        grid_dim: int = 30,
        num_train_pairs: int = 10,
        num_colors: int = 11,
        num_layers: int = 8,
        num_heads: int = 8,
        d_model: int = 512,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super(ARCTransformer, self).__init__()
        self.grid_dim = grid_dim
        self.num_train_pairs = num_train_pairs
        self.num_classes = num_colors + 1  # Add padding class
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        self.embedding = nn.Embedding(self.num_classes, self.d_model)
        self.pos_encoding = TrainablePositionalEncoding(
            self.d_model,
            max_len=(self.num_train_pairs * 2 + 1) * self.grid_dim * self.grid_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Enable gradient checkpointing
        self.transformer_encoder.use_checkpoint = True

        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self,
        grids: torch.Tensor,
        masks: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ):
        batch_size, num_grids, height, width = grids.size()

        # Apply grid masks
        masked_grids = torch.where(masks, grids, torch.zeros_like(grids))

        # Embed input
        x = self.embedding(
            masked_grids
        )  # This should now be (batch_size, num_grids, height, width, d_model)

        # Flatten grids
        x = x.view(batch_size, num_grids, -1, self.d_model)

        # Add trainable positional encoding
        x = self.pos_encoding(x)

        # Flatten the grid_masks to create attention mask
        mask = masks.view(batch_size, num_grids, -1).bool()

        # Create attention mask for transformer
        seq_len = (self.num_train_pairs * 2 + 1) * self.grid_dim * self.grid_dim
        attn_mask = mask.view(batch_size, -1)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, seq_len, -1)
        attn_mask = ~attn_mask

        # Instead of expanding for all heads at once, iterate over each head
        attn_mask_final = []
        for _ in range(self.num_heads):
            attn_mask_final.append(attn_mask)
        attn_mask = torch.stack(
            attn_mask_final
        )  # Shape: (num_heads, batch_size, seq_len, seq_len)
        attn_mask = attn_mask.reshape(self.num_heads * batch_size, seq_len, seq_len)

        key_padding_mask = ~mask.view(batch_size, -1)

        # Transformer encoder
        x = x.view(batch_size, -1, self.d_model)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(
            x, src_key_padding_mask=key_padding_mask, mask=attn_mask
        )

        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model)

        # Output layer
        x = self.output_layer(x)

        # Reshape to match the original grid shape
        x = x.view(batch_size, num_grids, height, width, self.num_classes)

        # If we're in training mode and output is provided, use teacher forcing
        if self.training and output is not None:
            output_embedded = self.embedding(output)
            x[:, -1] = self.output_layer(output_embedded).view(
                batch_size, height, width, -1
            )

        return x[:, -1]  # Return only the last grid (test output)


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(TrainablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1) * x.size(2)  # num_grids * (height * width)
        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        pos_encodings = self.pos_embedding(positions).view(
            x.size(0), x.size(1), x.size(2), -1
        )
        return x + pos_encodings
