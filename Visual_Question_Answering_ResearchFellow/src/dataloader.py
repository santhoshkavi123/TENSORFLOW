# Basic packages
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings

warnings.filterwarnings("ignore")
from typing import List, Dict
from PIL import Image
import requests
from tqdm import tqdm
import pickle

# ds packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

# Tensorflow packages
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_text import BertTokenizer, WhitespaceTokenizer


# Read modules from utils
from utils import load_text_data, tokenize_dict


# ================= Load all question pairs ================= #
train_QA_dict = load_text_data(
    file_path="./data/data_VQA/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt",
    sep="|",
)

print("---- Success : Load all question pairs ----")


# ================== Find out the maximum length of questions and answers ================== #
tokenizer = WhitespaceTokenizer()
max_length_question = 0
for record in tqdm(train_QA_dict):
    question = record["question"]
    tokens = tokenizer.tokenize(question)
    max_length_question = max(max_length_question, tokens.shape[0])

print(f"---- Max Length Question: {max_length_question} ----")

max_length_answer = 0
for record in tqdm(train_QA_dict):
    answer = record["answer"]
    tokens = tokenizer.tokenize(answer)
    max_length_answer = max(max_length_answer, tokens.shape[0])

print(f"---- Max Length Answer: {max_length_answer} ----")

max_length = {"Question": max_length_question, "Answer": max_length_answer}

try:
    with open("./data/processed/train/max_length.pkl", "wb") as f:
        pickle.dump(max_length, f)
    print("---- Success : Save max length ----")
except:
    print("---- Failed : Save max length ----")


# ====================== Tokenizer Initialization ====================== #
try:
    filepath = "./vocab.txt"
    tokenizer = BertTokenizer(filepath, lower_case=True, token_out_type=tf.int32)
    
    print("---- Success : Tokenizer Initialization ----")
except:
    print("---- Failed : Tokenizer Initialization ----")


try:
    train_QA_tokenized = tokenize_dict(
        data_dict=train_QA_dict, tokenizer=tokenizer, max_length=max_length
    )
    with open("./data/processed/train/train_QA_tokenized.pkl", "wb") as f:
        pickle.dump(train_QA_tokenized, f)

    print("---- Success : Tokenization ----")
except:
    print("---- Failed : Tokenization ----")


# ====================== Find all unique question words ====================== #
all_list_question = []
for records in tqdm(train_QA_dict):
    all_list_question.append(records["question"])

all_list_question = " ".join(all_list_question)

max_question_vocab_size = len(set(all_list_question.split(" ")))
print(f"Max Question Vocabulary Size : {max_question_vocab_size}")


# ====================== Find all unique answer words ======================= #
all_list_answer = []
for records in tqdm(train_QA_dict):
    all_list_answer.append(records["answer"])

all_list_answer = " ".join(all_list_answer)

max_answer_vocab_size = len(set(all_list_answer.split(" ")))
print(f"Max Answer Vocabulary Size : {max_answer_vocab_size}")


max_vocab_size = {
    "max_question_vocab_size": max_question_vocab_size,
    "max_answer_vocab_size": max_answer_vocab_size,
}

try:
    with open("./data/processed/train/max_vocab_size.pkl", "wb") as f:
        pickle.dump(max_vocab_size, f)

    print("---- Success : Save max vocab size ----")
except:
    print(" ---- Failed : Save max vocab size ----")


# Create Validation QA dictionary
try:
    valid_QA_dict = load_text_data(
        file_path="./data/data_VQA/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt",
        sep="|",
    )
    print("---- Success : Load Validation QA Dictionary ---- ")
except:
    print("---- Failed : Load Validation QA Dictionary ---- ")

try:
    with open("./data/processed/valid/valid_QA_dict.pkl", "wb") as f:
        pickle.dump(valid_QA_dict, f)
        print("---- Success : Save Validation QA Dictionary ----- ")
except:
    print("---- Failed : Save Validation QA Dictionary ---- ")


# Tokenize Validation QA pairs
try:
    valid_QA_tokenized = tokenize_dict(
        data_dict=valid_QA_dict, tokenizer=tokenizer, max_length=max_length
    )
    with open("./data/processed/valid/valid_QA_tokenized.pkl", "wb") as f:
        pickle.dump(valid_QA_tokenized, f)

    print("---- Success : Tokenization ----")
except:
    print("---- Failed : Tokenization ----")
