import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Verileri yükleme ve ön işleme
data = pd.read_csv("veri.csv")

def bootstrap_data(data, num_samples):
    sampled_data = data.sample(n=num_samples, replace=True)
    combined_data = pd.concat([data, sampled_data], ignore_index=True)
    return combined_data

data = bootstrap_data(data, 500)  # 500 rastgele örnek ekleyin

sorular = data["Soru"]
cevaplar = data["Cevap"]

# Tokenizer'ı oluşturma ve metin verilerini işleme
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sorular)
word_index = tokenizer.word_index

total_words = len(tokenizer.word_index) + 1

tokenized_sentences = tokenizer.texts_to_sequences(sorular.astype(str))

max_sequence_len = max([len(x) for x in tokenized_sentences])
input_sequences = pad_sequences(tokenized_sentences, maxlen=max_sequence_len, padding="pre")

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]

# Modeli oluşturma
model = Sequential()
model.add(Embedding(total_words, 40, input_length=max_sequence_len - 1))
model.add(Bidirectional(GRU(250, return_sequences=True)))
model.add(Bidirectional(LSTM(250)))
model.add(Dense(total_words, activation="softmax"))

# Modeli derleme
optimizer = Adam(learning_rate=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto')

# Modeli eğitme
model.fit(xs, labels, epochs=50, batch_size=240, verbose=1, callbacks=[earlystop])

# Modeli kaydetme
model.save("chatbot_model.keras")
