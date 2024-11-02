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

        x = x + self.dropout1(x1)
        x = self.norm1(x)

        x1 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x1)
        x = self.norm2(x)

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
        x = x + x_sa
        x = self.norm1(x)

        x_mha, mha_attn_weights = self._mha_block(
            x,
            memory,
            memory_mask,
            memory_key_padding_mask,
            is_causal=memory_is_causal,
            need_weights=need_weights,
        )
        x = x + x_mha
        x = self.norm2(x)

        x = x + self._ff_block(x)
        x = self.norm3(x)

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

        encoder_layer = EncoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )

        self.encoder = EncoderWithAttention(encoder_layer, self.num_encoder_layers)

        decoder_layer = DecoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )

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
        batch_size = src.shape[0]

        src = self.embedding.forward(src)

        pos_emb = self.pos_encoding.forward(src)
        src.add_(pos_emb)

        src = src.view(batch_size, self.seq_len, self.d_model)

        padding_mask = ~src_mask.view(batch_size, -1)

        memory, encoder_attn_weights = self.encoder.forward(
            src, src_key_padding_mask=padding_mask, need_weights=need_weights
        )

        output_query = self.output_query.expand(batch_size, -1, -1)

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

        output = self.output_layer(output)

        output = output.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

        return (
            output,
            encoder_attn_weights,
            decoder_sa_attn_weights,
            decoder_mha_attn_weights,
        )

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
            return (
                torch.argmax(output, dim=-1),
                encoder_attn_weights,
                decoder_sa_attn_weights,
                decoder_mha_attn_weights,
            )


class ARCTransformerEncoder(nn.Module):
    grid_dim: int
    num_train_pairs: int
    num_classes: int
    num_layers: int
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
        self.num_layers = params.num_encoder_layers
        self.num_heads = params.num_heads
        self.d_ff = params.d_ff
        self.dropout = params.dropout

        self.input_seq_len = (
            (self.num_train_pairs * 2 + 1) * self.grid_dim * self.grid_dim
        )
        self.output_seq_len = self.grid_dim * self.grid_dim
        self.seq_len = self.input_seq_len + self.output_seq_len

        self.embedding = nn.Embedding(self.num_classes, self.d_model)
        self.pos_encoding = ARCPositionalEncoding(
            d_model=self.d_model,
            grid_dim=self.grid_dim,
            num_train_pairs=self.num_train_pairs,
        )

        encoder_layer = EncoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )

        self.encoder = EncoderWithAttention(encoder_layer, self.num_layers)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     self.d_model, self.num_heads, self.d_ff, self.dropout, batch_first=True
        # )
        # self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.output_query = nn.Parameter(
            torch.randn(1, 1, self.grid_dim, self.grid_dim, self.d_model)
        )
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        batch_size = src.shape[0]

        embedded_src = self.embedding.forward(src)

        if tgt is not None:
            output_query = self.embedding.forward(tgt).view(
                batch_size, 1, self.grid_dim, self.grid_dim, self.d_model
            )
        else:
            output_query = self.output_query.expand(batch_size, -1, -1, -1, -1)

        combined_input = torch.cat([embedded_src, output_query], dim=1)

        # Add positional encodings
        pos_emb = self.pos_encoding(combined_input)
        embedded = combined_input + pos_emb

        embedded = embedded.view(batch_size, self.seq_len, self.d_model)

        causal_mask = torch.zeros(self.seq_len, self.seq_len, device=src.device)
        causal_mask[: self.input_seq_len, self.input_seq_len :] = 1
        causal_mask = causal_mask.bool()

        # Create padding mask
        padding_mask = ~src_mask.view(batch_size, -1)

        padding_mask = torch.cat(
            [
                padding_mask,
                torch.zeros(
                    (batch_size, self.grid_dim**2), dtype=torch.bool, device=src.device
                ),
            ],
            dim=1,
        )

        output = self.encoder.forward(
            embedded, mask=causal_mask, src_key_padding_mask=padding_mask
        )[0]

        # Get only the output grid portion
        output_grid_portion = output[:, -self.output_seq_len :, :]

        # Project to vocabulary space
        logits = self.output_layer(output_grid_portion)

        # Reshape to grid format
        output = logits.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

        if temperature > 0:
            output = output / temperature

        return (output, None, None, None)

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
            ) = self.forward(src, src_mask)
            return (
                torch.argmax(output, dim=-1),
                encoder_attn_weights,
                decoder_sa_attn_weights,
                decoder_mha_attn_weights,
            )


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolutional layer for patch embedding
        self.conv_embed = nn.Conv2d(
            in_channels=self.num_classes,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        x = (
            F.one_hot(x.long(), num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        x = self.conv_embed(x)

        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.embed_dim)

        mask = nn.functional.avg_pool2d(
            mask.float(),
            self.patch_size,
            stride=self.patch_size,
        )

        mask = (mask > 0).reshape(batch_size, -1)

        return (x, mask)


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
        self.patch_grid_dim = self.grid_dim // self.patch_size
        self.seq_len = num_grids * self.patch_grid_dim * self.patch_grid_dim

        self.embedding = PatchEmbedding(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embed_dim=self.d_model,
        )
        self.pos_encoding = ARCPositionalEncoding(
            d_model=self.d_model,
            grid_dim=self.patch_grid_dim,
            num_train_pairs=self.num_train_pairs,
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

        # Apply patch embedding
        src_patches, mask_patches = self.embedding.forward(src, src_mask)

        # Apply positional encoding
        pos_emb_patches = self.pos_encoding.forward(
            src_patches.reshape(
                batch_size,
                num_grids,
                self.patch_grid_dim,
                self.patch_grid_dim,
                self.d_model,
            )
        )
        pos_emb_patches = pos_emb_patches.reshape(-1, self.d_model)

        src_patches = src_patches.reshape(batch_size, -1, self.d_model)
        src_patches = src_patches + pos_emb_patches

        # Invert padding mask
        padding_mask = ~mask_patches

        # Encode input
        memory, encoder_attn_weights = self.encoder.forward(
            src_patches, src_key_padding_mask=padding_mask, need_weights=need_weights
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

        # Generate output patches
        output = self.output_layer.forward(output)

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


class ARCVisionEncoder(nn.Module):
    def __init__(self, params: ARCTransformerEncoderDecoderParams):
        super().__init__()
        self.grid_dim = params.grid_dim
        self.num_train_pairs = params.num_train_pairs
        self.num_classes = params.num_colors + 1
        self.d_model = params.d_model
        self.num_layers = params.num_encoder_layers
        self.num_heads = params.num_heads
        self.d_ff = params.d_ff
        self.dropout = params.dropout
        self.patch_size = 2

        self.patch_grid_dim = self.grid_dim // self.patch_size

        self.input_seq_len = (
            (self.num_train_pairs * 2 + 1) * self.patch_grid_dim * self.patch_grid_dim
        )
        self.output_seq_len = self.grid_dim * self.grid_dim
        self.seq_len = self.input_seq_len + self.output_seq_len

        self.embedding = PatchEmbedding(
            num_classes=self.num_classes,
            patch_size=self.patch_size,
            embed_dim=self.d_model,
        )
        self.pos_encoding = ARCPositionalEncodingV2(
            d_model=self.d_model,
            grid_dim=self.grid_dim,
            num_train_pairs=self.num_train_pairs,
        )

        encoder_layer = EncoderLayerWithAttention(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )
        self.encoder = EncoderWithAttention(encoder_layer, self.num_layers)

        self.output_query = nn.Parameter(
            torch.randn(1, 1, self.grid_dim, self.grid_dim, self.d_model)
        )
        self.output_layer = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, num_grids, grid_dim, grid_dim = src.shape

        pos_emb = self.pos_encoding.forward(
            num_grids=num_grids + 1, grid_dim=grid_dim, device=src.device
        )

        src = src.reshape(batch_size, num_grids * grid_dim, grid_dim)
        src_mask = src_mask.reshape(batch_size, num_grids * grid_dim, grid_dim)

        src_patched, mask_patched = self.embedding.forward(src, src_mask)

        input_pos_emb = pos_emb[:-1, :, :, :]
        input_pos_emb_patched = (
            input_pos_emb.unfold(1, self.patch_size, self.patch_size)
            .unfold(2, self.patch_size, self.patch_size)
            .mean(dim=(-2, -1))
        )

        src_patched = src_patched.reshape(batch_size, -1, self.d_model)
        input_pos_emb_patched = input_pos_emb_patched.reshape(-1, self.d_model)
        input_seq = src_patched + input_pos_emb_patched

        # if tgt is not None:
        #     output_query = self.embedding.forward(tgt).view(
        #         batch_size, 1, self.grid_dim, self.grid_dim, self.d_model
        #     )
        # else:
        output_query = self.output_query.expand(batch_size, -1, -1, -1, -1)
        output_pos_emb = pos_emb[-1:, :, :, :]

        output_query = output_query.reshape(batch_size, -1, self.d_model)
        output_pos_emb = output_pos_emb.reshape(-1, self.d_model)
        output_seq = output_query + output_pos_emb

        combined_seq = torch.cat([input_seq, output_seq], dim=1)

        # Make causal mask
        causal_mask = torch.zeros(self.seq_len, self.seq_len, device=src.device)
        causal_mask[: self.input_seq_len, self.input_seq_len :] = 1
        causal_mask = causal_mask.bool()

        # Make padding mask
        padding_mask = ~mask_patched
        padding_mask = torch.cat(
            [
                padding_mask,
                torch.zeros(
                    (batch_size, self.grid_dim**2), dtype=torch.bool, device=src.device
                ),
            ],
            dim=1,
        )

        output, attn_weights = self.encoder.forward(
            combined_seq,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
            need_weights=need_weights,
        )

        output_grid_portion = output[:, -self.output_seq_len :, :]

        logits = self.output_layer.forward(output_grid_portion)

        output = logits.view(batch_size, self.grid_dim, self.grid_dim, self.num_classes)

        if temperature > 0:
            output = output / temperature

        return (output, attn_weights)

    def generate(
        self, src: torch.Tensor, src_mask: torch.Tensor, need_weights: bool = False
    ):
        with torch.no_grad():
            (
                output,
                encoder_attn_weights,
            ) = self.forward(src, src_mask)
        return (
            torch.argmax(output, dim=-1),
            encoder_attn_weights,
        )


class ARCPositionalEncodingV2(nn.Module):
    def __init__(
        self,
        d_model: int,
        grid_dim: int,
        num_train_pairs: int,
    ):
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

    @torch.compiler.disable
    def forward(
        self, num_grids: int, grid_dim: int, device: torch.device
    ) -> torch.Tensor:
        grid_pos = torch.arange(grid_dim, device=device)

        # Row pos embedding
        row_emb = (
            self.row_embedding.forward(grid_pos)
            .unsqueeze(1)
            .expand(num_grids, -1, grid_dim, -1)
        )

        # Column pos embedding
        col_emb = (
            self.col_embedding.forward(grid_pos)
            .unsqueeze(0)
            .expand(num_grids, grid_dim, -1, -1)
        )

        # Input/output embedding
        grid_indices = torch.arange(num_grids, device=device)
        is_output = (grid_indices % 2 == 1).long()
        io_emb = (
            self.input_output_embedding(is_output)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(num_grids, grid_dim, grid_dim, -1)
        )

        # Pair embedding
        pair_indices = torch.div(grid_indices, 2, rounding_mode="floor")
        pair_emb = (
            self.pair_embedding(pair_indices)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(num_grids, grid_dim, grid_dim, -1)
        )

        # Combine all embeddings (1, num_grids, height, width, d_model)
        combined_emb = torch.cat([row_emb, col_emb, io_emb, pair_emb], dim=-1)

        return combined_emb


class ARCPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        grid_dim: int,
        num_train_pairs: int,
    ):
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
