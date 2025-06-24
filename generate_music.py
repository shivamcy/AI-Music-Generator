import os
import numpy as np
import pickle
import tempfile
import soundfile as sf
from tensorflow.keras.models import load_model
from music21 import stream, note, chord
from pretty_midi import PrettyMIDI

def convert_midi_to_wav(midi_path):
    midi_data = PrettyMIDI(midi_path)
    audio = midi_data.synthesize()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, audio, 44100, format='WAV')
        return f.name

def generate_music(sequence_length=50, num_notes=100):
    os.makedirs("generated_music", exist_ok=True)
    midi_output_path = os.path.abspath("generated_music/output.mid")

    with open("models/notes.pkl", "rb") as f:
        notes = pickle.load(f)

    unique_notes = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(unique_notes)}
    int_to_note = {i: n for i, n in enumerate(unique_notes)}

    model = load_model("models/music_generator.h5")

    start_idx = np.random.randint(len(notes) - sequence_length)
    sequence_input = [note_to_int[n] for n in notes[start_idx : start_idx + sequence_length]]
    generated_notes = []

    for _ in range(num_notes):
        input_data = np.array(sequence_input).reshape(1, sequence_length, 1).astype(np.float32)
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_note = int_to_note[predicted_index]
        generated_notes.append(predicted_note)
        sequence_input.append(predicted_index)
        sequence_input = sequence_input[1:]

    midi_stream = stream.Stream()

    for n in generated_notes:
        if "." in n or n.isdigit():
            chord_notes = [note.Note(int(num)) for num in n.split('.')]
            midi_stream.append(chord.Chord(chord_notes))
        else:
            midi_stream.append(note.Note(n))

    midi_stream.write("midi", fp=midi_output_path)
    print(f"🎼 MIDI saved to {midi_output_path}")

    wav_path = convert_midi_to_wav(midi_output_path)
    print(f"🔊 WAV saved to {wav_path}")
    return wav_path

if __name__ == "__main__":
    generate_music()
