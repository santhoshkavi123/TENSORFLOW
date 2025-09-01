import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Conv1D,
    GlobalAveragePooling1D,
    Dense,
    Add,
)


class QuestionAttention(Model):
    def __init__(self, vocab_size):
        super(QuestionAttention, self).__init__()

        # Embedding layer
        self.embedding_layer = tf.keras.layers.Embedding(
            vocab_size, 128, name="embedding_layer"
        )

        # LSTM layer
        self.lstm = Bidirectional(LSTM(256, return_sequences=True))

        # Convolutional layers
        self.conv_layer_1 = Conv1D(
            filters=128, kernel_size=3, padding="same", activation="relu"
        )
        self.conv_layer_1 = Conv1D(
            filters=256, kernel_size=3, padding="same", activation="relu"
        )
        self.conv_layer_2 = Conv1D(
            filters=512, kernel_size=5, padding="same", activation="relu"
        )

        # Global Average Pooling (1D since it's sequences, not 2D images)
        self.global_avg_pool = GlobalAveragePooling1D()

        # Output layer 
        self.output_layer = Dense(256, activation="softmax")

    def call(self, inputs):
        # Get BERT embeddings
        # x = self.encoder(tf.squeeze(inputs, axis = 1))["last_hidden_state"]  # (batch, seq_len, hidden)
        x = self.embedding_layer(inputs)

        # LSTM layer
        lstm_output = self.lstm(x)

        # Convolutional layers
        x = self.conv_layer_1(lstm_output)
        x = self.conv_layer_2(x)

        # Skip Connection
        x = Add()([lstm_output, x])

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Output layer
        out = self.output_layer(x)

        return out
