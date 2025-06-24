def generate_music_from_trained_model(num_notes=200):
    with open("model/note_distribution.pkl", "rb") as f:
        notes = pickle.load(f)

    new_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    start = 0
    duration = 0.5
    for _ in range(num_notes):
        pitch = random.choice(notes)
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start+duration)
        piano.notes.append(note)
        start += duration

    new_midi.instruments.append(piano)
    os.makedirs("generated_music", exist_ok=True)

    output_path = "generated_music/output.mid"
    new_midi.write(output_path)
    return output_path  # return .mid instead of .wav
