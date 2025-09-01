import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
import pickle

from tensorflow_text import BertTokenizer

class VQAEvaluator:
    def __init__(self, model, start_token=1, end_token=2, max_len=15):
        """
        model: trained VQA model
        tokenizer: tokenizer with detokenize method
        start_token: start token ID for answers
        end_token: end token ID for answers
        max_len: maximum length of predicted answer
        """
        self.model = model
        self.start_token = start_token
        self.end_token = end_token
        self.max_len = max_len

    def greedy_decode(self, img, ques):
        """Greedy decoding for answer generation."""
        ans_seq = [self.start_token]

        for _ in range(self.max_len):
            decoder_in = tf.expand_dims(ans_seq, axis=0)  # (1, cur_len)
            logits = self.model(img, ques, decoder_in, training=False)  # (1, cur_len, vocab_size)
            next_token = tf.argmax(logits[:, -1, :], axis=-1).numpy()[0]

            ans_seq.append(next_token)
            if next_token == self.end_token:
                break

        return ans_seq


    def visualize_random_sample(self, dataset, num_samples=1):
        """Pick random samples from dataset, run prediction, and visualize results."""
        all_batches = list(dataset)  # collect all batches into memory
        batch = random.choice(all_batches)
        (img_batch, ques_batch), ans_batch = batch
        print(batch)

        for i in range(num_samples):
            idx = random.randint(0, img_batch.shape[0] - 1)

            img = img_batch[idx:idx+1]      # keep batch dim
            ques = ques_batch[idx:idx+1]
            ans_true = ans_batch[idx].numpy()

            # Predict
            pred_tokens = self.greedy_decode(img, ques)
            pred_text = BertTokenizer().detokenize(pred_tokens)
            true_text = BertTokenizer().detokenize(ans_true)
            ques_text = BertTokenizer().detokenize(ques.numpy()[0])

            # Plot
            plt.imshow(img[0])
            plt.axis("off")
            plt.title("Visual Question Answering")
            plt.show()

            print("Question:", ques_text)
            print("True Answer:", true_text)
            print("Predicted Answer:", pred_text)
            print("="*60)
