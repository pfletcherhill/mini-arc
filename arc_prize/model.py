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

        self.embedding = nn.Embedding(self.num_classes, self.d_model)
        self.pos_encoding = ARCPositionalEncoding(
            d_model=self.d_model,
            grid_dim=self.grid_dim,
            num_train_pairs=self.num_train_pairs,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)

        self.output_query = nn.Parameter(torch.randn(1, self.grid_dim**2, self.d_model))
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(self, src, src_mask):
        # src shape: (batch_size, num_input_grids, grid_dim, grid_dim)
        batch_size = src.shape[0]

        src = self.embedding(
            src
        )  # (batch_size, num_input_grids * grid_dim * grid_dim, d_model)

        # Add positional encoding
        src = self.pos_encoding(src)

        # Flatten grids
        src = src.view(
            batch_size, -1, self.d_model
        )  # (batch_size, num_input_grids * grid_dim * grid_dim)

        # Encode input
        padding_mask = ~src_mask.view(batch_size, -1)
        # print("size padding mask", padding_mask.shape)
        # print("src size", src.shape, src.transpose(0, 1).shape)
        # visualize_mask(padding_mask, "Padding mask")

        memory = self.encoder(src.transpose(0, 1), src_key_padding_mask=padding_mask)

        # Prepare output query
        output_query = self.output_query.expand(batch_size, -1, -1).transpose(0, 1)

        # Decode
        output = self.decoder(output_query, memory)

        # Generate output grid
        output = self.output_layer(output.transpose(0, 1))

        # Reshape to grid
        return output.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

    def generate(self, src, src_mask=None):
        with torch.no_grad():
            output = self.forward(src, src_mask)
            # print("output shape", output.shape)
            # print("output sample", output[0, 0, 0])
            return torch.argmax(output, dim=-1)


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
