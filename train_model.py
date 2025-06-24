import pretty_midi
import os
import pickle

def train_model_on_midis(midi_paths):
    notes = []
    for midi_path in midi_paths:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes.extend([note.pitch for note in instrument.notes])

    # Simple dummy model: just save the note distribution
    os.makedirs("model", exist_ok=True)
    with open("model/note_distribution.pkl", "wb") as f:
        pickle.dump(notes, f)
