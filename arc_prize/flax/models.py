import jax
import jax.numpy as jnp
from flax import nnx, struct


@struct.dataclass
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


def sinusoidal_encoding(length: int, dim: int) -> jnp.ndarray:
    position = jnp.arange(length)
    div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim))
    sin_enc = jnp.sin(position[:, None] * div_term)
    cos_enc = jnp.cos(position[:, None] * div_term)
    encoding = jnp.concatenate([sin_enc, cos_enc], axis=-1)
    return encoding[:, :dim]  # Ensure the output dimension is correct


def create_arc_fixed_positional_encoding(
    d_model: int, grid_dim: int, num_train_pairs: int
) -> jnp.ndarray:
    assert d_model % 4 == 0, "d_model must be divisible by 4"
    embed_dim = d_model // 4

    # Row and column encodings
    row_enc = sinusoidal_encoding(grid_dim, embed_dim)
    col_enc = sinusoidal_encoding(grid_dim, embed_dim)

    # Input/output encoding
    # Using sinusoidal encoding for I/O to maintain consistency and to fully utilize the allocated dimensions
    io_enc = sinusoidal_encoding(2, embed_dim)

    # Pair encoding
    pair_enc = sinusoidal_encoding(num_train_pairs + 1, embed_dim)  # +1 for test pair

    # Combine encodings
    def apply_encoding(grid_idx: int) -> jnp.ndarray:
        is_output = grid_idx % 2
        pair_idx = grid_idx // 2

        grid_encoding = jnp.concatenate(
            [
                row_enc[:, None, :].repeat(grid_dim, axis=1),
                col_enc[None, :, :].repeat(grid_dim, axis=0),
                jnp.broadcast_to(io_enc[is_output], (grid_dim, grid_dim, embed_dim)),
                jnp.broadcast_to(pair_enc[pair_idx], (grid_dim, grid_dim, embed_dim)),
            ],
            axis=-1,
        )

        return grid_encoding

    return jax.vmap(apply_encoding)(jnp.arange(2 * num_train_pairs + 1))


class MlpBlock(nnx.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(
            in_features=d_model,
            out_features=d_ff,
            rngs=rngs,
        )
        self.linear2 = nnx.Linear(
            in_features=d_ff,
            out_features=d_model,
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, inputs: jax.Array, *, rngs: nnx.Rngs | None = None):
        x = self.linear1(inputs)
        x = nnx.relu(x)
        x = self.dropout(x, rngs=rngs)
        output = self.linear2(x)
        output = self.dropout(output, rngs=rngs)
        return output


class EncoderLayer(nnx.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float, rngs: nnx.Rngs
    ):
        self.ln1 = nnx.LayerNorm(
            num_features=d_model,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(
            num_features=d_model,
            rngs=rngs,
        )
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.mlp_block = MlpBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout, rngs=rngs
        )
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self,
        inputs: jax.Array,
        *,
        masks: jax.Array | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        if masks is not None:
            batch_size, input_len, _ = inputs.shape
            masks = nnx.make_attention_mask(
                masks, jnp.ones((batch_size, input_len), dtype=jnp.bool)
            )
        # Self attention block
        x = self.ln1(inputs)
        x = self.self_attention(x, mask=masks, rngs=rngs)
        x = self.dropout(x, rngs=rngs)
        x = x + inputs
        # MLP block
        z = self.ln2(x)
        z = self.mlp_block(z, rngs=rngs)
        return x + z


class DecoderLayer(nnx.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float, rngs: nnx.Rngs
    ):
        self.ln1 = nnx.LayerNorm(
            num_features=d_model,
            rngs=rngs,
        )
        self.ln2 = nnx.LayerNorm(
            num_features=d_model,
            rngs=rngs,
        )
        self.ln3 = nnx.LayerNorm(
            num_features=d_model,
            rngs=rngs,
        )
        self.self_attention = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=d_model, dropout_rate=dropout, rngs=rngs
        )
        self.multihead_attention = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=d_model, dropout_rate=dropout, rngs=rngs
        )
        self.mlp_block = MlpBlock(
            d_model=d_model, d_ff=d_ff, dropout=dropout, rngs=rngs
        )
        self.dropout1 = nnx.Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self,
        tgt: jax.Array,
        memory: jax.Array,
        *,
        tgt_masks: jax.Array | None = None,
        memory_masks: jax.Array | None = None,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        batch_size, tgt_len, _ = tgt.shape
        batch_size, memory_len, _ = memory.shape

        self_attention_masks = None
        if tgt_masks is not None:
            self_attention_masks = nnx.make_attention_mask(
                tgt_masks, jnp.ones((batch_size, tgt_len), dtype=jnp.bool)
            )

        multihead_attention_masks = None
        if memory_masks is not None or tgt_masks is not None:
            multihead_attention_masks = nnx.make_attention_mask(
                tgt_masks
                if tgt_masks is not None
                else jnp.ones((batch_size, tgt_len), dtype=jnp.bool),
                memory_masks
                if memory_masks is not None
                else jnp.ones((batch_size, memory_len), dtype=jnp.bool),
            )

        # Self attention block
        x = self.ln1(tgt)
        x = self.self_attention(x, mask=self_attention_masks, rngs=rngs)
        x = self.dropout1(x, rngs=rngs)
        x = x + tgt

        # Multihead attention block
        y = self.ln2(x)
        y = self.multihead_attention(
            y, memory, memory, mask=multihead_attention_masks, rngs=rngs
        )
        y = self.dropout2(y, rngs=rngs)
        y = y + x

        # MLP block
        z = self.ln3(y)
        z = self.mlp_block(z, rngs=rngs)
        return y + z


class ARCTransformerEncoderDecoder(nnx.Module):
    def __init__(self, params: ARCTransformerEncoderDecoderParams, rngs: nnx.Rngs):
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

        self.embedding = nnx.Embed(
            num_embeddings=self.num_classes, features=self.d_model, rngs=rngs
        )
        self.encoder_layers = [
            EncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                rngs=rngs,
            )
            for _ in range(self.num_encoder_layers)
        ]
        self.decoder_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                rngs=rngs,
            )
            for _ in range(self.num_decoder_layers)
        ]

        self.output_query = nnx.Param(
            jax.random.uniform(rngs.params(), (1, self.grid_dim**2, self.d_model))
        )

        self.output_layer = nnx.Linear(
            in_features=self.d_model, out_features=self.num_classes, rngs=rngs
        )

    def __call__(
        self,
        src: jnp.ndarray,
        src_mask: jnp.ndarray,
        rngs: nnx.Rngs | None = None,
    ) -> jnp.ndarray:
        batch_size = src.shape[0]

        src = self.embedding(src)

        pos_encoding = create_arc_fixed_positional_encoding(
            self.d_model, self.grid_dim, self.num_train_pairs
        )

        src = src + pos_encoding[None, :, :, :, :]

        src = src.reshape(batch_size, self.seq_len, self.d_model)

        src_mask = src_mask.reshape(batch_size, -1)

        memory = src
        for layer in self.encoder_layers:
            memory = layer(memory, masks=src_mask, rngs=rngs)

        output = jnp.broadcast_to(
            self.output_query.value, (batch_size, self.grid_dim**2, self.d_model)
        )
        for layer in self.decoder_layers:
            output = layer(output, memory, memory_masks=src_mask, rngs=rngs)

        output = self.output_layer(output)

        output = output.reshape(
            batch_size, self.grid_dim, self.grid_dim, self.num_classes
        )

        return output

    def generate(self, src: jnp.ndarray, src_mask: jnp.ndarray) -> jnp.ndarray:
        output = self(src, src_mask)

        return jnp.argmax(output, axis=-1)
