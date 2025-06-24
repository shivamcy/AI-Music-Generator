import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

def extract_notes_from_midi(midi_paths):
    # Dummy implementation — replace with real MIDI note parsing logic
    notes = []
    for path in midi_paths:
        notes += ["C4", "E4", "G4"] * 50  # fake notes
    os.makedirs("models", exist_ok=True)
    with open("models/notes.pkl", "wb") as f:
        pickle.dump(notes, f)

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

def train_model_on_midis(midi_paths):
    extract_notes_from_midi(midi_paths)
    X, y, vocab_size = load_data()
    X = np.expand_dims(X, -1)
    model = build_model(vocab_size)
    model.fit(X, y, epochs=10, batch_size=64)
    model.save("models/music_generator.h5")
    print("✅ Training complete and model saved.")

if __name__ == "__main__":
    # test with dummy files
    train_model_on_midis(["example1.mid", "example2.mid", "example3.mid"])
