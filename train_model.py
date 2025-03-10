import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_data():
    with open("models/notes.pkl", "rb") as f:
        notes = pickle.load(f)

    unique_notes = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(unique_notes)}

    sequence_length = 50
    input_sequences = []
    output_notes = []

    for i in range(len(notes) - sequence_length):
        input_sequences.append([note_to_int[n] for n in notes[i:i + sequence_length]])
        output_notes.append(note_to_int[notes[i + sequence_length]])

    input_sequences = np.array(input_sequences)
    output_notes = to_categorical(output_notes, num_classes=len(unique_notes))

    return input_sequences, output_notes, len(unique_notes)

def build_model(n_vocab):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(50, 1)),
        Dropout(0.2),
        LSTM(128),
        Dense(128, activation="relu"),
        Dense(n_vocab, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model

if __name__ == "__main__":
    X, y, vocab_size = load_data()
    X = np.expand_dims(X, -1)  # Reshape for LSTM

    model = build_model(vocab_size)
    model.fit(X, y, epochs=50, batch_size=64)

    model.save("models/music_generator.h5")
    print("🎵 Model Training Complete! Saved as 'models/music_generator.h5'.")
