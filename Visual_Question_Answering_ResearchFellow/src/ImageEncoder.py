import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


class ImageEncoder(Model):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.base_model = ResNet50(
            include_top=False, weights="imagenet", input_shape=(64, 64, 3)
        )

        self.base_model.trainable = False
        # Global Average Pooling layer
        self.global_avg_pool = GlobalAveragePooling2D()

        # Output layer
        self.output_layer = Dense(256, activation = "softmax")

    def call(self, inputs):
        x = self.base_model(inputs)

        # Global average pooling to flatten the features
        x = self.global_avg_pool(x)

        # Output layer
        x = self.output_layer(x)

        return x
