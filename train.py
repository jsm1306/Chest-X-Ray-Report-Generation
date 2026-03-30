# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow import keras
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import KFold
import cv2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Concatenate, Dropout
from tensorflow.keras.layers import Lambda, RepeatVector
from tensorflow.keras.models import Model
import re
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
 #   for filename in filenames:
  #      print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_report = pd.read_csv('/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv')
df_pro = pd.read_csv('/kaggle/input/chest-xrays-indiana-university/indiana_projections.csv')
df_pro = df_pro.drop_duplicates(subset="uid").reset_index(drop=True)
df_report = df_report.drop_duplicates(subset="uid").reset_index(drop=True)

common_uids = set(df_pro["uid"]).intersection(set(df_report["uid"]))

df_pro = df_pro[df_pro["uid"].isin(common_uids)].sort_values("uid").reset_index(drop=True)
df_report = df_report[df_report["uid"].isin(common_uids)].sort_values("uid").reset_index(drop=True)

print("Images shape =", df_pro.shape)
print("Reports shape =", df_report.shape)
image_path = "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/"
images = glob(image_path + "*.png")
resized_images = []
mean = 0.485  
std = 0.229

for img_name in df_pro["filename"]:
    img_path = os.path.join(image_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Warning: Image not found - {img_name}")
        continue
    
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = (img - mean) / std
    img = np.expand_dims(img, axis=-1)
    resized_images.append(img)

resized_images = np.array(resized_images)
print("Final shape:", resized_images.shape)

cnn_encoder = models.Sequential([ 
    layers.Input(shape=(224, 224, 1)), 
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)), layers.Conv2D(256, (3,3), activation='relu', padding='same'),
    
    layers.Conv2D(256, (3,3), activation='relu', padding='same'), 
    layers.Conv2D(256, (3,3), activation='relu', padding='same'), 
    layers.MaxPooling2D((2,2)), layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    
    layers.Conv2D(512, (3,3), activation='relu', padding='same'), 
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)), 
    
    layers.Conv2D(512, (3,3), activation='relu', padding='same'), 
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.Conv2D(512, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)) ])
input_layer = layers.Input(shape=(224, 224, 1))
x = cnn_encoder(input_layer)
x = layers.Reshape((49, 512))(x)
encoder_model = models.Model(inputs=input_layer, outputs=x)
encoder_model.summary()
encoder_features = encoder_model.predict(resized_images, batch_size=32, verbose=1)
print(encoder_features.shape)
#(3851, 49, 512)

reports = df_report["findings"].astype(str).tolist() 
reports = [r if len(str(r).strip()) > 0 else "no findings" for r in reports]
def clean_text(text):
    text = text.lower()                                  
    text = re.sub(r"[^a-z0-9\s.,]", "", text)            
    text = re.sub(r"\s+", " ", text).strip()            
    return text
df_report["findings"] = df_report["findings"].astype(str)
df_report["findings"] = df_report["findings"].apply(lambda x: clean_text(x))
df_report["findings"] = df_report["findings"].apply(lambda x: x if len(x.strip()) > 0 else "no findings")
df_report["findings"] = df_report["findings"].apply(lambda x: "<start> " + x + " <end>")
reports = df_report["findings"].tolist()
print("Total reports:", len(reports))
# for getting unique qords so that i can change vocab size accordingly
tokenizer_test = Tokenizer(oov_token="<unk>")
tokenizer_test.fit_on_texts(df_report["findings"])

print("Actual unique words in dataset:", len(tokenizer_test.word_index))
vocab_size =2000

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
tokenizer.fit_on_texts(df_report["findings"])
report_sequences = tokenizer.texts_to_sequences(df_report["findings"])
max_len = 200

# Decoder input = all words except last
decoder_input_data = pad_sequences([seq[:-1] for seq in report_sequences], 
                                   maxlen=max_len-1, padding='post')

# Decoder output = all words except first
decoder_output_data = pad_sequences([seq[1:] for seq in report_sequences], 
                                    maxlen=max_len-1, padding='post')

# For sparse_categorical_crossentropy
decoder_output_data = np.expand_dims(decoder_output_data, -1)
report_padded = pad_sequences(report_sequences, maxlen=max_len, padding='post', truncating='post')
# prepare decoder unput and output
decoder_input  = report_padded[:, :-1]
decoder_output = report_padded[:, 1:]
encoder_input = layers.Input(shape=(49, 512), name='encoder_input')   # extracted CNN features
decoder_input = layers.Input(shape=(max_len - 1,), name='decoder_input')

embedding = layers.Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True, name='embedding')(decoder_input)

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, encoder_output, hidden_state):
        hidden_state_with_time_axis = tf.expand_dims(hidden_state, 1)
        score = tf.nn.tanh(self.W1(encoder_output) + self.W2(hidden_state_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
# Define encoder and decoder inputs
encoder_input = layers.Input(shape=(49, 512), name='encoder_input')
decoder_input = layers.Input(shape=(max_len - 1,), name='decoder_input')

# Embedding layer for decoder
embedding = layers.Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True, name='embedding')(decoder_input)

# Initialize decoder LSTM
decoder_lstm = layers.LSTM(512, return_sequences=True, return_state=True, name='decoder_lstm')

# Attention layer
attention = BahdanauAttention(256)

# Process decoder step-by-step
# (we use Lambda or custom loop for simplicity demonstration)
def apply_attention(inputs):
    encoder_out, decoder_emb = inputs
    hidden_state = tf.reduce_mean(encoder_out, axis=1)  # dummy hidden state
    attention = BahdanauAttention(512)
    context_vector, attn_weights = attention(encoder_out, hidden_state)

    # Expand and repeat context to match decoder time steps
    context_vector = tf.expand_dims(context_vector, 1)
    context_vector = tf.repeat(context_vector, repeats=tf.shape(decoder_emb)[1], axis=1)

    combined = tf.concat([context_vector, decoder_emb], axis=-1)
    return combined


context_combined = layers.Lambda(apply_attention, output_shape=(max_len - 1, 256 + 512))(
    [encoder_input, embedding]
)

# Pass through decoder LSTM
decoder_outputs, _, _ = decoder_lstm(context_combined)

# Output layer
output = layers.Dense(vocab_size, activation='softmax', name='output')(decoder_outputs)


encoder_features = encoder_features.astype('float32')
decoder_input_data = decoder_input_data.astype('int32')
decoder_output_data = decoder_output_data.astype('int32')

# Define number of folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train_index, val_index in kfold.split(encoder_features):
    print(f"\n Training Fold {fold_no} ------------------------------")

    # Split the data for this fold
    X_train_enc, X_val_enc = encoder_features[train_index], encoder_features[val_index]
    X_train_dec_inp, X_val_dec_inp = decoder_input_data[train_index], decoder_input_data[val_index]
    y_train, y_val = decoder_output_data[train_index], decoder_output_data[val_index]

    # Build your model from scratch (important!)
    full_model = models.Model(inputs=[encoder_input, decoder_input], outputs=output)

    full_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model for this fold
    history = full_model.fit(
        [X_train_enc, X_train_dec_inp], y_train,
        validation_data=([X_val_enc, X_val_dec_inp], y_val),
        epochs=20,
        batch_size=32,
        verbose=1
    )
def generate_report(img_path, encoder_model, full_model, tokenizer, max_len=200, mean=0.485, std=0.229):
    """
    Robust inference for your trained `full_model`.
    - max_len: the original `max_len` you used (e.g., 200)
    - the model expects decoder_input shape (max_len - 1,)
    """
    import cv2, numpy as np, re
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # 1) read + preprocess (must match training preprocessing)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = (img - mean) / std
    img = np.expand_dims(img, axis=[0, -1])  # shape (1,224,224,1), as used for encoder

    # 2) get encoder features (shape (1,49,512) for your encoder)
    enc_feats = encoder_model.predict(img, verbose=0)

    # 3) safe token ids - many tokenizers use different names; try multiple fallbacks
    start_token = (tokenizer.word_index.get('<start>') or
                   tokenizer.word_index.get('startseq') or
                   tokenizer.word_index.get('sos') or
                   tokenizer.word_index.get('start') or
                   1)  # last-resort
    end_token = (tokenizer.word_index.get('<end>') or
                 tokenizer.word_index.get('endseq') or
                 tokenizer.word_index.get('eos') or
                 tokenizer.word_index.get('end') or
                 2)

    # 4) prepare sequence for generation
    decoder_maxlen = max_len - 1  # because you trained decoder with max_len-1
    input_seq = [start_token]
    result_words = []

    for step in range(decoder_maxlen):
        # pad to the SAME length used during training
        dec_inp = pad_sequences([input_seq], maxlen=decoder_maxlen, padding='post')

        # predict full sequence logits; take the logits at the next position index
        preds = full_model.predict([enc_feats, dec_inp], verbose=0)  # shape (1, decoder_maxlen, vocab)
        # next token position is len(input_seq)-1
        pos = len(input_seq) - 1
        if pos >= preds.shape[1]:
            # safety: if position beyond model output, stop
            break
        probs = preds[0, pos]  # (vocab_size,)

        # greedy selection (argmax)
        next_id = int(np.argmax(probs))

        # stop if end token
        if next_id == end_token:
            break

        # skip padding and the start token
        if next_id == 0 or next_id == start_token:
            # if model outputs pad/start, try to continue (but avoid infinite loop)
            # append a small safeguard; here we skip appending and continue
            input_seq.append(next_id)
            continue

        word = tokenizer.index_word.get(next_id, "")
        if word:
            result_words.append(word)
        input_seq.append(next_id)

    # 5) format and clean
    if not result_words:
        return "Findings:\n Unable to generate report."

    report = " ".join(result_words)
    # Only remove token markers like <unk>, <start>, <end>; avoid removing legitimate words
    report = re.sub(r"(<unk>|<start>|<end>)", " ", report)
    report = re.sub(r"\s+", " ", report).strip()
    if not report.endswith("."):
        report += "."

    # Capitalize first char
    report = report[0].upper() + report[1:]

    return "Findings:\n" + report

generated = generate_report(
    "/kaggle/input/chest-xrays-indiana-university/images/images_normalized/1000_IM-0003-2001.dcm.png",
    encoder_model,  # your encoder model object
    full_model,     # the trained fold model you want to use for inference
    tokenizer,
    max_len=200
)
print(generated)
