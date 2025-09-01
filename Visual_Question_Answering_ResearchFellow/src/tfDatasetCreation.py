import os
import warnings

warnings.filterwarnings("ignore")


from typing import List, Dict
from utils import preprocess_image, create_tf_dataset
import tensorflow as tf
import pickle

# Create Train Tensorflow Datasets
try:
    with open("./data/processed/train/train_QA_tokenized.pkl", "rb") as f:
        train_QA_tokenized = pickle.load(f)
    print("---- Success : Load Train QA Tokenized ----")
except:
    print("---- Failed : Load Train QA Tokenized ----")


try:
    train_tf = create_tf_dataset(
        tokenized_data=train_QA_tokenized,
        image_dir="./data/data_VQA/ImageClef-2019-VQA-Med-Training/Train_images/",
        batch_size=32,
    )
    save_path = "./data/processed/tf_dataset/tf_train"
    tf.data.Dataset.save(train_tf, save_path)
    print("---- Success : Create Train TF Dataset ----")
except:
    print("---- Failed : Create Train TF Dataset -----")


# Create Validation Tensorflow Datasets
try:
    with open("./data/processed/valid/valid_QA_tokenized.pkl", "rb") as f:
        valid_QA_tokenized = pickle.load(f)
    print("---- Success : Load Valid QA Tokenized ----")
except:
    print("---- Failed : Load Valid QA Tokenized ----")


try:
    valid_tf = create_tf_dataset(
        tokenized_data=valid_QA_tokenized,
        image_dir="./data/data_VQA/ImageClef-2019-VQA-Med-Validation/Val_images/",
        batch_size=32,
    )
    save_path = "./data/processed/tf_dataset/tf_valid"
    tf.data.Dataset.save(valid_tf, save_path)
    print("---- Success : Create Valid TF Dataset ----")
except:
    print("---- Failed : Create Valid TF Dataset -----")
