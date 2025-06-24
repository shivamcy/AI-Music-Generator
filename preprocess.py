from music21 import converter, note, chord
import glob
import pickle

def preprocess_midi_files(data_path="midi_files/*.mid"):
    notes = []

    for file in glob.glob(data_path):
        midi = converter.parse(file)
        print(f"Processing {file}...")

        for part in midi.parts:
            for element in part.recurse():
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    print(f"Extracted {len(notes)} notes")
    with open("models/notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    return notes

if __name__ == "__main__":
    preprocess_midi_files()
