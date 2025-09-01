import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding,
    MultiHeadAttention,
    LayerNormalization,
    Dense,
    Dropout,
)


def create_look_ahead_mask(seq_len):
    """Create a mask to mask future tokens in the decoder."""
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask  # 1s where future tokens are masked

class AnswerDecoder(Model):
    def __init__(
        self,
        embedding_dim,
        hidden_dim_attn,
        vocabulary_size,
        num_heads_attn,
        ff_dim,
        dropout_rate=0.3,
    ):
        super(AnswerDecoder, self).__init__()

        # Answer embedding layer
        self.answer_embedding = Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
        )

        # Decoder layers
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads_attn, key_dim=hidden_dim_attn
        )
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads_attn, key_dim=hidden_dim_attn
        )
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.norm3 = LayerNormalization()

        # Feed Forward networks
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(hidden_dim_attn),
            ]
        )

        self.dropout = Dropout(dropout_rate)

        # Output projection to vocabulary
        self.output_layer = Dense(vocabulary_size, activation = "softmax") 

    def call(self, decoder_input, encoder_feat, training=False):

        # Embedding
        x = self.answer_embedding(decoder_input)

        # Masked Self-Attention
        seq_len = tf.shape(x)[1]
        look_ahead_mask = create_look_ahead_mask(seq_len)[
            tf.newaxis, tf.newaxis, :, :
        ]  # add batch & head dims

        self_attn_output = self.self_attn(
            query=x, value=x, key=x, attention_mask=look_ahead_mask
        )
        x = self.norm1(self_attn_output + x)

        # Cross-Attention with encoder features
        cross_attn_output = self.cross_attn(
            query=x, value=encoder_feat, key=encoder_feat
        )
        x = self.norm2(cross_attn_output + x)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(ffn_output + x)
        x = self.dropout(x, training=training)

        logits = self.output_layer(x)  # (batch, ans_seq_len, vocab_size)
        return logits
