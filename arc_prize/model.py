import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ARCTransformerEncoderDecoderParams:
    grid_dim: int
    num_train_pairs: int
    num_colors: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    dropout: float


class ARCTransformerEncoderDecoder(nn.Module):
    grid_dim: int
    num_train_pairs: int
    num_classes: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    dropout: float
    seq_len: int

    def __init__(self, params: ARCTransformerEncoderDecoderParams):
        super().__init__()
        self.grid_dim = params.grid_dim
        self.num_train_pairs = params.num_train_pairs
        self.num_classes = params.num_colors + 1
        self.d_model = params.d_model
        self.num_encoder_layers = params.num_encoder_layers
        self.num_decoder_layers = params.num_decoder_layers
        self.num_heads = params.num_heads
        self.d_ff = params.d_ff
        self.dropout = params.dropout
        self.seq_len = (self.num_train_pairs * 2 + 1) * self.grid_dim * self.grid_dim

        self.embedding = nn.Embedding(self.num_classes, self.d_model)
        self.pos_encoding = ARCPositionalEncoding(
            d_model=self.d_model,
            grid_dim=self.grid_dim,
            num_train_pairs=self.num_train_pairs,
        )
        # self.pos_encoding = HybridARCPositionalEncoding(
        #     d_model=self.d_model,
        #     grid_dim=self.grid_dim,
        #     num_train_pairs=self.num_train_pairs,
        # )
        # self.pos_encoding = TrainablePositionalEncoding(
        #     self.d_model,
        #     max_len=((self.num_train_pairs * 2 + 1) * self.grid_dim * self.grid_dim),
        # )
        # self.pos_encoding = SinusoidalPositionalEncoding(
        #     self.d_model,
        #     max_len=((self.num_train_pairs * 2 + 1) * self.grid_dim * self.grid_dim),
        # )

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)

        self.output_query = nn.Parameter(torch.randn(1, self.grid_dim**2, self.d_model))
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(self, src, src_mask):
        # src shape: (batch_size, num_input_grids, grid_dim, grid_dim)
        batch_size = src.shape[0]

        src = self.embedding(src)

        # Add positional encoding
        src = self.pos_encoding(src)

        # Flatten grids
        src = src.view(batch_size, self.seq_len, self.d_model)

        # Encode input
        padding_mask = ~src_mask.view(batch_size, -1)
        # print("size padding mask", padding_mask.shape)
        # print("src size", src.shape, src.transpose(0, 1).shape)
        # visualize_mask(padding_mask, "Padding mask")

        memory = self.encoder(src, src_key_padding_mask=padding_mask)

        # Prepare output query
        output_query = self.output_query.expand(batch_size, -1, -1)

        # Decode
        output = self.decoder(output_query, memory)

        # Generate output grid
        output = self.output_layer(output)

        # Reshape to grid
        return output.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

    def generate(self, src, src_mask=None):
        with torch.no_grad():
            output = self.forward(src, src_mask)
            # print("output shape", output.shape)
            # print("output sample", output[0, 0, 0])
            return torch.argmax(output, dim=-1)


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

    def forward(self, x):
        """
        x: tensor of shape (batch_size, num_grids, height, width, d_model)
        """
        batch_size, num_grids, height, width, _ = x.size()

        # Create position indices
        row_pos = torch.arange(height, device=x.device).unsqueeze(1).expand(-1, width)
        col_pos = torch.arange(width, device=x.device).unsqueeze(0).expand(height, -1)

        # Get embeddings for row and column positions
        row_emb = self.row_embedding(row_pos)
        col_emb = self.col_embedding(col_pos)

        # Combine row and column embeddings
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)

        # Expand to match input shape
        pos_emb = (
            pos_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, num_grids, -1, -1, -1)
        )

        # Create grid indices tensor
        grid_indices = (
            torch.arange(num_grids, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        # Determine input/output based on even/odd index
        is_output = (grid_indices % 2 == 1).long()
        io_emb = self.input_output_embedding(is_output)

        # Determine pair index (integer division by 2)
        pair_indices = torch.div(grid_indices, 2, rounding_mode="floor")
        # Set the last grid (test input) to have a separate pair index
        pair_indices[:, -1] = self.num_train_pairs
        pair_emb = self.pair_embedding(pair_indices)

        # Expand io_emb and pair_emb to match grid dimensions
        io_emb = io_emb.unsqueeze(2).unsqueeze(2).expand(-1, -1, height, width, -1)
        pair_emb = pair_emb.unsqueeze(2).unsqueeze(2).expand(-1, -1, height, width, -1)

        # Combine all embeddings
        combined_emb = torch.cat([pos_emb, io_emb, pair_emb], dim=-1)

        return x + combined_emb


class HybridARCPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, grid_dim: int, num_train_pairs: int):
        super().__init__()
        self.d_model = d_model
        self.grid_dim = grid_dim
        self.num_train_pairs = num_train_pairs

        # Sinusoidal encodings for row and column positions
        self.pos_encoding = self.create_sinusoidal_encoding(grid_dim, d_model // 2)

        # Learned embeddings for input/output and pair index
        self.input_output_embedding = nn.Embedding(2, d_model // 4)
        self.pair_embedding = nn.Embedding(num_train_pairs + 1, d_model // 4)

    def create_sinusoidal_encoding(self, length, dim):
        encoding = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding

    def forward(self, x: torch.Tensor):
        print("forward", x.shape)
        batch_size, num_grids, height, width, _ = x.size()

        print("sin encoding", self.pos_encoding.shape, self.pos_encoding)

        # Apply sinusoidal positional encoding
        row_pos = self.pos_encoding[:height, :].unsqueeze(1).expand(-1, width, -1)
        col_pos = self.pos_encoding[:width, :].unsqueeze(0).expand(height, -1, -1)
        print("row, col", row_pos.shape, col_pos.shape)
        pos_emb = torch.cat([row_pos, col_pos], dim=-1)
        pos_emb = (
            pos_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, num_grids, -1, -1, -1)
        )
        print("pos_emb shape", pos_emb.shape)

        # Apply learned embeddings for input/output and pair index (same as before)
        grid_indices = (
            torch.arange(num_grids, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )
        print("grid_indices", grid_indices.shape, grid_indices)
        is_output = (grid_indices % 2 == 1).long()
        io_emb = self.input_output_embedding(is_output)
        pair_indices = torch.div(grid_indices, 2, rounding_mode="floor")
        pair_indices[:, -1] = self.num_train_pairs
        pair_emb = self.pair_embedding(pair_indices)

        io_emb = io_emb.unsqueeze(2).unsqueeze(2).expand(-1, -1, height, width, -1)
        pair_emb = pair_emb.unsqueeze(2).unsqueeze(2).expand(-1, -1, height, width, -1)

        print("io and pair emb", io_emb.shape, pair_emb.shape)

        # Combine all embeddings
        combined_emb = torch.cat([pos_emb, io_emb, pair_emb], dim=-1)
        print("comb emb", combined_emb.shape)

        return x + combined_emb


class TrainablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(TrainablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        batch_size, num_grids, height, width, d_model = x.size()
        seq_len = num_grids * height * width

        positions = (
            torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        pos_encodings = self.pos_embedding(positions).view(
            batch_size, num_grids, height, width, -1
        )
        return x + pos_encodings


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Create a buffer to store the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        batch_size, num_grids, height, width, d_model = x.size()
        seq_len = num_grids * height * width

        # Ensure we have enough positional encodings
        assert (
            seq_len <= self.max_len
        ), f"Sequence length {seq_len} exceeds maximum length {self.max_len}"

        # Get the positional encodings for the required sequence length
        pos_encodings = self.pe[:, :seq_len]

        # Reshape to match the input dimensions
        pos_encodings = pos_encodings.view(1, num_grids, height, width, d_model)

        # Expand to match the batch size
        pos_encodings = pos_encodings.expand(batch_size, -1, -1, -1, -1)

        return x + pos_encodings
