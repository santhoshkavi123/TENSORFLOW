import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


# Neural Network Architecture
from ImageEncoder import ImageEncoder
from QuestionEncoder import QuestionAttention
from DecoderArchitecture import AnswerDecoder

@keras.saving.register_keras_serializable()
class VQATransformer(Model):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim_attn=256, num_heads_attn=8, ff_dim=512, dropout_rate=0.3):
        super(VQATransformer, self).__init__()

        # Image encoder
        self.image_encoder = ImageEncoder()

        # Question encoder
        self.question_encoder = QuestionAttention(vocab_size=vocab_size["max_question_vocab_size"])

        # Answer decoder
        self.answer_decoder = AnswerDecoder(
            embedding_dim=embedding_dim,
            hidden_dim_attn=hidden_dim_attn,
            vocabulary_size=vocab_size["max_answer_vocab_size"],
            num_heads_attn=num_heads_attn,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        )
        
        # Final layer (optional fusion if needed)
        self.fc = Dense(vocab_size["max_answer_vocab_size"], activation="softmax")

    def call(self, image_inputs, question_inputs, answer_inputs, training=False):
        # 1. Encode Image
        img_features = self.image_encoder(image_inputs)       # (batch, 256)

        # 2. Encode Question
        ques_features = self.question_encoder(question_inputs) # (batch, 256)

        # 3. Concatenate image + question features
        encoder_features = tf.concat([img_features, ques_features], axis=-1)  # (batch, 512)

        # Expand dims so cross-attn sees it as sequence
        encoder_features = tf.expand_dims(encoder_features, axis=1)           # (batch, 1, 512)

        # 4. Decode Answer sequence (teacher forcing during training)
        logits = self.answer_decoder(answer_inputs, encoder_features, training=training) # (batch, seq_len, vocab_size)

        return logits
