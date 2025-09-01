import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")
from typing import List, Dict

# ds packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

# Tensorflow packages
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load Text Data
def load_text_data(file_path: str, sep: str) -> List[Dict]:
    text_data = pd.read_csv(file_path, sep=sep, header=None)
    text_data.columns = ["image_id", "question", "answer"]
    print(f"Shape of Text Data : {text_data.shape}")

    text_data_dict = text_data.to_dict(
        orient="records"
    )  # Convert to a list of dictionaries
    return text_data_dict


# Preprocess Image
def preprocess_image(image_id, image_dir):

    # load the image
    image_filename = image_id + ".jpg"
    image_path = os.path.join(image_dir, image_filename)

    # basic preprocessing the image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = img / 255.0
    img = tf.cast(img, tf.float32)

    return img


# Tokenizer Helper function
def tokenize_dict(data_dict: List[Dict], tokenizer, max_length: Dict) -> List[Dict]:

    data_dict_tokenized = []

    for record in tqdm(data_dict):

        # Question Tokenization
        question = record["question"]
        question_tokens = tokenizer.tokenize(question).to_list()
        question_tokens = list(
            chain.from_iterable(chain.from_iterable(question_tokens))
        )
        question_tokens = pad_sequences(
            [question_tokens],
            maxlen=max_length["Question"] + 20,
            dtype="int32",
            padding="post",
        )

        # Answer Tokenization
        answer = record["answer"]
        answer_tokens = tokenizer.tokenize(answer).to_list()
        answer_tokens = list(chain.from_iterable(chain.from_iterable(answer_tokens)))
        answer_tokens = pad_sequences(
            [answer_tokens],
            maxlen=max_length["Answer"] + 20,
            dtype="int32",
            padding="post",
        )

        data_dict_tokenized.append(
            {
                "image_id": record["image_id"],
                "question": question_tokens,
                "answer": answer_tokens,
            }
        )

    return data_dict_tokenized


# tensorflow dataset creation
def create_tf_dataset(tokenized_data: List[Dict], image_dir: str, batch_size: int):

    def gen():
        for record in tqdm(tokenized_data):
            img = preprocess_image(
                image_id=record["image_id"],
                image_dir=image_dir,
            )
            question = record["question"].reshape(-1)
            answer = record["answer"].reshape(-1)

            yield (img, question), answer

    # Infer shapes/dtypes for TF
    output_signature = (
        (
            tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float32),  # Example image shape
            tf.TensorSpec(shape=(31,), dtype=tf.int32),  # Question tokens
        ),
        tf.TensorSpec(shape=(39,), dtype=tf.int32),  # Answer token
    )

    dataset = (
        tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset
