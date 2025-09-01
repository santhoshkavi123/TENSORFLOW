import argparse
import tensorflow as tf
import pickle
from VQATransformer import VQATransformer
from tensorflow.keras.models import Model
from Predictor import VQAEvaluator
from tensorflow_text import BertTokenizer
import numpy as np
import keras
import matplotlib.pyplot as plt

# ----------------------------
# Argument Parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train VQA Transformer")
    parser.add_argument("--input", type=str, required=True, help="Path to training data (.pkl)")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab size pickle")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--train", type=str, default="artifacts/vqa_model.keras", help="Path to save best model")
    return parser.parse_args()

# ----------------------------
# Training Model Wrapper
# ----------------------------
@keras.saving.register_keras_serializable()
class VQAModel(tf.keras.Model):
    def __init__(self, vqa_transformer, loss_fn, **kwargs):
        super().__init__(**kwargs)
        self.vqa_transformer = vqa_transformer
        self.loss_fn = loss_fn

    def train_step(self, data):
        (img, ques), ans = data
        decoder_in = ans[:, :-1]
        target_out = ans[:, 1:]

        with tf.GradientTape() as tape:
            logits = self.vqa_transformer(img, ques, decoder_in, training=True)
            loss = self.loss_fn(target_out, logits)

        grads = tape.gradient(loss, self.vqa_transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqa_transformer.trainable_variables))
        return {"loss": loss}

    def test_step(self, data):
        (img, ques), ans = data
        decoder_in = ans[:, :-1]
        target_out = ans[:, 1:]

        logits = self.vqa_transformer(img, ques, decoder_in, training=False)
        loss = self.loss_fn(target_out, logits)
        return {"loss": loss}

# ----------------------------
# Main Function
# ----------------------------
def main():
    args = parse_args()

    # Load vocab size
    with open(args.vocab, "rb") as f:
        vocab_size = pickle.load(f)

    # Load training data
    train_tf = tf.data.Dataset.load(args.input)

    valid_tf = tf.data.Dataset.load("./data/processed/tf_dataset/tf_valid")

    # Load model
    model = VQATransformer(vocab_size=vocab_size)
    print("Model Summary:")
    model.summary()

    # Compile wrapped model
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    vqa_wrapper = VQAModel(model, loss_fn)
    vqa_wrapper.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

    # Callbacks
    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath=args.train,
    #         monitor="val_loss",
    #         save_best_only=True,
    #         verbose=1,
    #         save_weights_only=True 
    #     ),
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor="val_loss",
    #         patience=3,
    #         restore_best_weights=True
    #     )
    # ]

    print("Starting training ....")
    # Train
    history =  vqa_wrapper.fit(
        train_tf,
        epochs=args.epochs,
        validation_data = valid_tf,
    )
    try:
        vqa_wrapper.save(args.train)
        print("---- Success : Model saved ----")
    except:
        print("---- Failed : Model Saved ----")

    print("Training complete.")

    print("Sample Evaluation")

    with open("./data/processed/train/max_length.pkl", "rb") as f:
        max_length = pickle.load(f)


    def greedy_decode(model, img, ques, start_token, end_token, max_len=15):
        """
        model: VQATransformer
        img: (1, 64, 64, 3)
        ques: (1, q_len)
        start_token: int (ID for <sos>)
        end_token: int (ID for <eos>)
        """
        ans_seq = [start_token]

        for _ in range(max_len):
            decoder_in = tf.expand_dims(ans_seq, axis=0)  # (1, cur_len)

            logits = model(img, ques, decoder_in, training=False)  # (1, cur_len, vocab_size)
            next_token = tf.argmax(logits[:, -1, :], axis=-1).numpy()[0]

            ans_seq.append(next_token)
            if next_token == end_token:
                break

        return ans_seq
    
   # Take one sample
    for (img_batch, ques_batch), ans_batch in valid_tf.take(1):
        img = img_batch[0:1]        # (1, 64, 64, 3)
        ques = ques_batch[0:1]      # (1, q_len)
        ans_true = ans_batch[0].numpy()
        break

    # Define start/end tokens (adapt to your tokenizer)
    START_TOKEN, END_TOKEN = 1, 2  

    pred_tokens = greedy_decode(model, img, ques, START_TOKEN, END_TOKEN, max_len=max_length["Answer"])

    pred_tokens = np.array([int(t) for t in pred_tokens])
    # Convert tokens → words
    tokenizer = BertTokenizer("./vocab.txt", 
                          lower_case = True, 
                          token_out_type= tf.int32)
    pred_text = tokenizer.detokenize(pred_tokens.reshape((1, -1)))
    true_text = tokenizer.detokenize(ans_true.reshape((1, -1)))

    plt.imshow(img[0])
    plt.axis("off")
    plt.title("VQA Sample")
    plt.show()

    print("Question:", tokenizer.detokenize(ques[0].numpy().reshape((1, -1))))
    print("True Answer:", true_text)
    print("Predicted Answer:", pred_text)

    
if __name__ == "__main__":
    main()
