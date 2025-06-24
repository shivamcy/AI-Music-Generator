import pretty_midi
import numpy as np
import os

def midi_to_note_sequence(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))
    notes.sort()  # Sort by start time
    return notes

def extract_features_from_directory(directory):
    all_notes = []
    for filename in os.listdir(directory):
        if filename.endswith(".mid"):
            path = os.path.join(directory, filename)
            notes = midi_to_note_sequence(path)
            pitches = [note[2] for note in notes]
            all_notes.extend(pitches)
    return np.array(all_notes)
