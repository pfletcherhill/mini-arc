import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass(frozen=True)
class ARCTransformerMaskedEncoderDecoderParams:
    grid_dim: int
    num_train_pairs: int
    num_colors: int
    num_grid_encoder_layers: int
    num_pair_encoder_layers: int
    num_global_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    dropout: float


class EncoderLayerWithAttention(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        x1, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(x1))
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))

        x = self.norm2(x + self.dropout2(x2))

        return x, attn_weights


class EncoderWithAttention(nn.TransformerEncoder):
    def __init__(
        self,
        encoder_layer: "EncoderLayerWithAttention",
        num_layers: int,
    ) -> None:
        super().__init__(encoder_layer, num_layers)

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        attn_weights = []

        for mod in self.layers:
            output, layer_attn_weights = mod(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                need_weights=need_weights,
            )
            if need_weights:
                attn_weights.append(layer_attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return output, (torch.stack(attn_weights, dim=1) if need_weights else None)


class DecoderLayerWithAttention(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, batch_first=True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = tgt
        x_sa, sa_attn_weights = self._sa_block(
            x,
            tgt_mask,
            tgt_key_padding_mask,
            is_causal=tgt_is_causal,
            need_weights=need_weights,
        )
        x = self.norm1(x + x_sa)

        x_mha, mha_attn_weights = self._mha_block(
            x,
            memory,
            memory_mask,
            memory_key_padding_mask,
            is_causal=memory_is_causal,
            need_weights=need_weights,
        )
        x = self.norm2(x + x_mha)
        x = self.norm3(x + self._ff_block(x))

        return x, sa_attn_weights, mha_attn_weights

    # self-attention block
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, sa_attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return self.dropout1(x), sa_attn_weights

    # multihead attention block
    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, mha_attn_weights = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return self.dropout2(x), mha_attn_weights

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderWithAttention(nn.TransformerDecoder):
    def __init__(
        self,
        decoder_layer: "DecoderLayerWithAttention",
        num_layers: int,
    ) -> None:
        super().__init__(decoder_layer, num_layers)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: Optional[bool] = False,
        memory_is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        output = tgt

        sa_attn_weights = []
        mha_attn_weights = []

        for mod in self.layers:
            output, layer_sa_attn_weights, layer_mha_attn_weights = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
                need_weights=need_weights,
            )
            if need_weights:
                sa_attn_weights.append(layer_sa_attn_weights)
                mha_attn_weights.append(layer_mha_attn_weights)

        if self.norm is not None:
            output = self.norm(output)

        return (
            output,
            torch.stack(sa_attn_weights, dim=1) if need_weights else None,
            torch.stack(mha_attn_weights, dim=1) if need_weights else None,
        )


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

        # encoder_layer = nn.TransformerEncoderLayer(
        #     self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        # )
        encoder_layer = EncoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )

        # self.encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        self.encoder = EncoderWithAttention(encoder_layer, self.num_encoder_layers)

        # decoder_layer = nn.TransformerDecoderLayer(
        #     self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        # )
        decoder_layer = DecoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )

        # self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)
        self.decoder = DecoderWithAttention(decoder_layer, self.num_decoder_layers)

        self.output_query = nn.Parameter(torch.randn(1, self.grid_dim**2, self.d_model))
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor, need_weights: bool = False
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # src shape: (batch_size, num_input_grids, grid_dim, grid_dim)
        batch_size = src.shape[0]

        src = self.embedding(src)

        # Add positional encoding
        src = self.pos_encoding(src)

        # Flatten grids
        src = src.view(batch_size, self.seq_len, self.d_model)

        # Encode input
        padding_mask = ~src_mask.view(batch_size, -1)

        memory, encoder_attn_weights = self.encoder.forward(
            src, src_key_padding_mask=padding_mask, need_weights=need_weights
        )

        # Prepare output query
        output_query = self.output_query.expand(batch_size, -1, -1)

        # Decode
        (
            output,
            decoder_sa_attn_weights,
            decoder_mha_attn_weights,
        ) = self.decoder.forward(
            output_query,
            memory,
            memory_key_padding_mask=padding_mask,
            need_weights=need_weights,
        )

        # Generate output grid
        output = self.output_layer(output)

        # Reshape to grid
        output = output.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

        return (
            output,
            encoder_attn_weights,
            decoder_sa_attn_weights,
            decoder_mha_attn_weights,
        )

    # TODO: potentially add temperature arg
    def generate(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        with torch.no_grad():
            (
                output,
                encoder_attn_weights,
                decoder_sa_attn_weights,
                decoder_mha_attn_weights,
            ) = self.forward(src, src_mask, need_weights)
            # print("output shape", output.shape)
            # print("output sample", output[0, 0, 0])
            return (
                torch.argmax(output, dim=-1),
                encoder_attn_weights,
                decoder_sa_attn_weights,
                decoder_mha_attn_weights,
            )


class ARCTransformerMaskedEncoderDecoder(nn.Module):
    grid_dim: int
    num_train_pairs: int
    num_classes: int
    num_grid_encoder_layers: int
    num_pair_encoder_layers: int
    num_global_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    d_model: int
    d_ff: int
    dropout: float
    seq_len: int

    def __init__(self, params: ARCTransformerMaskedEncoderDecoderParams):
        super().__init__()
        self.grid_dim = params.grid_dim
        self.num_train_pairs = params.num_train_pairs
        self.num_classes = params.num_colors + 1
        self.d_model = params.d_model
        self.num_grid_encoder_layers = params.num_grid_encoder_layers
        self.num_pair_encoder_layers = params.num_pair_encoder_layers
        self.num_global_encoder_layers = params.num_global_encoder_layers
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

        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        )
        num_total_encoder_layers = (
            self.num_grid_encoder_layers
            + self.num_pair_encoder_layers
            + self.num_global_encoder_layers
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_total_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_decoder_layers)

        self.output_query = nn.Parameter(torch.randn(1, self.grid_dim**2, self.d_model))
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        # src shape: (batch_size, num_input_grids, grid_dim, grid_dim)
        batch_size = src.shape[0]

        src = self.embedding(src)

        # Add positional encoding
        src = self.pos_encoding(src)

        # Flatten grids
        src = src.view(batch_size, self.seq_len, self.d_model)

        padding_mask = ~src_mask.view(batch_size, -1)
        # print("padding_mask", padding_mask.shape, padding_mask)
        # float_padding_mask = torch.zeros_like(padding_mask, dtype=torch.float)
        # float_padding_mask[padding_mask] = float("-inf")

        # print(
        #     "padding_mask",
        #     float_padding_mask.shape,
        #     float_padding_mask.min(),
        #     float_padding_mask.max(),
        # )

        attention_masks = []

        # Create grid attention mask
        grid_attention_mask = torch.ones(
            self.seq_len, self.seq_len, dtype=torch.bool, device=src.device
        )
        # grid_attention_mask = torch.full(
        #     size=(self.seq_len, self.seq_len),
        #     fill_value=float("-inf"),
        #     dtype=torch.float,
        # )
        for i in range(self.num_train_pairs * 2 + 1):
            start = i * self.grid_dim * self.grid_dim
            end = (i + 1) * self.grid_dim * self.grid_dim
            grid_attention_mask[start:end, start:end] = False
        for i in range(self.num_grid_encoder_layers):
            attention_masks.append(grid_attention_mask)

        # Create pair attention mask
        pair_attention_mask = torch.ones(
            self.seq_len, self.seq_len, dtype=torch.bool, device=src.device
        )
        # pair_attention_mask = torch.full(
        #     size=(self.seq_len, self.seq_len),
        #     fill_value=float("-inf"),
        #     dtype=torch.float,
        # )
        for i in range(self.num_train_pairs):
            start = (i * 2) * self.grid_dim * self.grid_dim
            end = (i + 1) * 2 * self.grid_dim * self.grid_dim
            pair_attention_mask[start:end, start:end] = False
        for i in range(self.num_pair_encoder_layers):
            attention_masks.append(pair_attention_mask)

        # Create global attention mask
        global_attention_mask = torch.zeros(
            self.seq_len, self.seq_len, dtype=torch.bool, device=src.device
        )
        # global_attention_mask = torch.full(
        #     size=(self.seq_len, self.seq_len), fill_value=0, dtype=torch.float
        # )
        for i in range(self.num_global_encoder_layers):
            attention_masks.append(global_attention_mask)

        memory = src
        # print("memory", memory.shape, "nan", torch.isnan(memory).sum())
        for i, encoder_layer in enumerate(self.encoder.layers):
            mask = attention_masks[i]
            mask_float = torch.zeros_like(mask, dtype=torch.float)
            mask_float[mask] = float(-1e9)

            padding_mask_float = torch.zeros_like(padding_mask, dtype=torch.float)
            padding_mask_float[padding_mask] = float(-1e9)
            # mask_expanded = mask.view(1, 1, self.seq_len, self.seq_len).expand(
            #     batch_size, self.num_heads, -1, -1
            # )
            # padding_mask_expanded = padding_mask.view(
            #     batch_size, 1, 1, self.seq_len
            # ).expand(-1, self.num_heads, -1, -1)
            # merged_mask = (mask_expanded + padding_mask_expanded).view(
            #     batch_size * self.num_heads, self.seq_len, self.seq_len
            # )
            # merged_mask_float = torch.zeros_like(merged_mask, dtype=torch.float64)
            # merged_mask_float[merged_mask] = float(-1e2)
            # print(
            #     "mask_float",
            #     mask_float.shape,
            #     mask_float.min(),
            #     mask_float.max(),
            # )

            # print("padding_mask", padding_mask.shape)

            # visualize_attention_mask(padding_mask, "Padding mask")

            # print(
            #     "merged_mask",
            #     merged_mask.shape,
            #     "true",
            #     merged_mask.sum(),
            #     "false",
            #     merged_mask.numel() - merged_mask.sum(),
            #     merged_mask[0][0],
            # )

            # visualize_attention_mask(mask_float, "Attention mask")

            # print("padding_mask_float", padding_mask_float.shape)

            # visualize_attention_mask(padding_mask_float, "Padding mask")

            memory = encoder_layer(
                memory, src_mask=mask_float, src_key_padding_mask=padding_mask_float
            )
            # print("layer", i, memory.shape, "nan count", torch.isnan(memory).sum())
            # visualize_nan_patterns(memory)
            # for i, batch in enumerate(memory):
            #     visualize_tensor(src[i].detach(), f"Before {i}")
            #     visualize_tensor(batch.detach(), f"After {i}")

        if self.encoder.norm is not None:
            memory = self.encoder.norm(memory)

        # for i, batch in enumerate(memory):
        #     print("nan count", torch.isnan(batch).sum())
        #     visualize_tensor(batch.detach(), f"After {i}")

        # Prepare output query
        output_query = self.output_query.expand(batch_size, -1, -1)

        # Decode
        output = self.decoder(output_query, memory)

        # Generate output grid
        output = self.output_layer(output)

        # Reshape to grid
        return output.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

    def generate(self, src, src_mask):
        with torch.no_grad():
            output = self.forward(src, src_mask)
            return torch.argmax(output, dim=-1)


class HybridPatchEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        num_train_pairs: int,
        embed_dim: int,
        grid_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_dim = grid_dim
        self.num_train_pairs = num_train_pairs

        # Color embedding
        self.color_embedding = nn.Embedding(self.num_classes, self.embed_dim)

        # Convolutional layer for patch embedding
        self.conv_embed = nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Positional embedding
        num_grids = self.num_train_pairs * 2 + 1
        self.num_patches = (num_grids * self.grid_dim // self.patch_size) * (
            self.grid_dim // self.patch_size
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.embed_dim)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, grid_size, grid_size)
        batch_size = x.shape[0]

        # Color embedding
        x = self.color_embedding(x)

        # Reshape for convolutional layer
        x = x.float().permute(0, 3, 1, 2)  # (batch_size, embed_dim, height, width)

        # Patch embedding using convolution
        x = self.conv_embed(x)

        # Reshape for positional embedding
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.embed_dim)

        # Add positional embedding
        x += self.pos_embedding

        # Patch mask

        mask = nn.functional.avg_pool2d(
            mask.float(),
            self.patch_size,
            stride=self.patch_size,
        )

        mask = (mask > 0).reshape(batch_size, -1)

        return (x, mask)


class ConcatPatchEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        num_train_pairs: int,
        embed_dim: int,
        grid_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_dim = grid_dim
        self.num_train_pairs = num_train_pairs

        # Color embedding
        self.color_embed_dim = embed_dim // 2
        self.color_embedding = nn.Embedding(self.num_classes, self.color_embed_dim)

        # Convolutional layer for patch embedding
        self.conv_embed = nn.Conv2d(
            in_channels=self.color_embed_dim,
            out_channels=(self.embed_dim - self.color_embed_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Positional embedding
        num_grids = self.num_train_pairs * 2 + 1
        self.num_patches = (num_grids * self.grid_dim // self.patch_size) * (
            self.grid_dim // self.patch_size
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.embed_dim)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, grid_size, grid_size)
        batch_size = x.shape[0]

        # Color embedding
        color_embed = self.color_embedding(x).float()

        # Reshape for convolutional layer
        x = color_embed.permute(0, 3, 1, 2)  # (batch_size, embed_dim, height, width)

        # Patch embedding using convolution
        patch_embed = self.conv_embed(x)

        # Reshape for positional embedding
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.embed_dim
        )

        combined_embed = torch.cat([color_embed, patch_embed])

        # Add positional embedding
        combined_embed += self.pos_embedding

        # Patch mask

        mask = nn.functional.avg_pool2d(
            mask.float(),
            self.patch_size,
            stride=self.patch_size,
        )

        mask = (mask > 0).reshape(batch_size, -1)

        return (combined_embed, mask)


class ARCVisionEncoderDecoder(nn.Module):
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
        self.patch_size = 2

        num_grids = self.num_train_pairs * 2 + 1
        self.seq_len = (num_grids * self.grid_dim // self.patch_size) * (
            self.grid_dim // self.patch_size
        )

        self.embedding = HybridPatchEmbedding(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            num_train_pairs=self.num_train_pairs,
            embed_dim=self.d_model,
            grid_dim=self.grid_dim,
        )

        encoder_layer = EncoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )
        self.encoder = EncoderWithAttention(encoder_layer, self.num_encoder_layers)

        decoder_layer = DecoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )
        self.decoder = DecoderWithAttention(decoder_layer, self.num_decoder_layers)

        self.output_query = nn.Parameter(
            torch.randn(1, self.grid_dim * self.grid_dim, self.d_model)
        )
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor, need_weights: bool = False
    ):
        batch_size, num_grids, grid_dim, grid_dim = src.shape

        # Flatten grids
        src = src.reshape(batch_size, num_grids * grid_dim, grid_dim)
        src_mask = src_mask.reshape(batch_size, num_grids * grid_dim, grid_dim)

        # Apply hybrid embedding
        src_patches, mask_patches = self.embedding.forward(src, src_mask)

        # Create padding mask

        padding_mask = ~mask_patches

        # Encode input
        memory, encoder_attn_weights = self.encoder(
            src_patches, src_key_padding_mask=padding_mask, need_weights=need_weights
        )

        # Prepare output query
        output_query = self.output_query.expand(batch_size, -1, -1)

        # Decode
        output, decoder_sa_attn_weights, decoder_mha_attn_weights = self.decoder(
            output_query,
            memory,
            memory_key_padding_mask=padding_mask,
            need_weights=need_weights,
        )

        # Generate output patches
        output = self.output_layer(output)

        # Reshape to grid
        output = output.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

        return (
            output,
            encoder_attn_weights,
            decoder_sa_attn_weights,
            decoder_mha_attn_weights,
        )

    def generate(
        self, src: torch.Tensor, src_mask: torch.Tensor, need_weights: bool = False
    ):
        with torch.no_grad():
            (
                output,
                encoder_attn_weights,
                decoder_sa_attn_weights,
                decoder_mha_attn_weights,
            ) = self.forward(src, src_mask, need_weights)
            return (
                torch.argmax(output, dim=-1),
                encoder_attn_weights,
                decoder_sa_attn_weights,
                decoder_mha_attn_weights,
            )


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
