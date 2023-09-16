import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Modeli yükle
model = load_model("chatbot_model.keras")

# Veri işleme araçlarını yükle
tokenizer = None
encoder = None

# Eğer tokenizasyon ve kodlama yapıldıysa, ilgili dosyaları yükleyin
try:
    import pickle
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
except FileNotFoundError:
    pass

while True:
    user_input = input("Soru: ")
    if user_input.lower() == 'quit':
        break
    
    if tokenizer is not None and encoder is not None:
        user_input_seq = tokenizer.texts_to_sequences([user_input])
        user_input_padded = pad_sequences(user_input_seq, maxlen=50, padding='post', truncating='post')
    
        predicted_probabilities = model.predict(user_input_padded)
        predicted_class = np.argmax(predicted_probabilities, axis=-1)
        predicted_answer = encoder.inverse_transform(predicted_class)
    
        print("Cevap:", predicted_answer[0])
    else:
        print("Model veya veri işleme araçları eksik. Lütfen eğitim işlemi tamamlanmadan bu kodu çalıştırmayın.")
