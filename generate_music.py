import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from music21 import stream, note, chord

def generate_music(output_file="generated_music/output.mid", sequence_length=50, num_notes=100):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load preprocessed notes
    with open("models/notes.pkl", "rb") as f:
        notes = pickle.load(f)

    unique_notes = sorted(set(notes))

    # Create mappings
    note_to_int = {n: i for i, n in enumerate(unique_notes)}
    int_to_note = {i: n for i, n in enumerate(unique_notes)}

    model = load_model("models/music_generator.h5")

    # Select a random starting sequence from the dataset
    start_idx = np.random.randint(len(notes) - sequence_length)
    sequence_input = notes[start_idx : start_idx + sequence_length]

    # Convert sequence to numerical format
    sequence_input = [note_to_int[n] for n in sequence_input]

    generated_notes = []

    for _ in range(num_notes):
        # Prepare input data for prediction
        input_data = np.array(sequence_input).reshape(1, sequence_length, 1)
        input_data = input_data.astype(np.float32)

        # Predict next note
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_note = int_to_note[predicted_index]

        # Append predicted note to output
        generated_notes.append(predicted_note)

        # Update input sequence
        sequence_input.append(predicted_index)
        sequence_input = sequence_input[1:]  # Keep length fixed

    # Convert generated notes to MIDI
    midi_stream = stream.Stream()

    for n in generated_notes:
        if "." in n or n.isdigit():
            chord_notes = [note.Note(int(num)) for num in n.split('.')]
            new_chord = chord.Chord(chord_notes)
            midi_stream.append(new_chord)
        else:
            midi_stream.append(note.Note(n))

    midi_stream.write("midi", fp=output_file)
    print(f"🎼 Music Generated! Saved as {output_file}")

if __name__ == "__main__":
    generate_music()
