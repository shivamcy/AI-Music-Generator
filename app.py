import gradio as gr
import os
import uuid
import shutil
from train_model import train_model_on_midis
from generate_music import generate_music  # returns .wav path

# Ensure folders exist
os.makedirs("user_midi", exist_ok=True)
os.makedirs("generated_music", exist_ok=True)

model_trained = False

def train_model(midi_files):
    global model_trained
    if len(midi_files) < 3:
        return gr.update(value="❌ Please upload at least 3 MIDI files."), False

    midi_paths = []
    for file in midi_files:
        filename = f"{uuid.uuid4().hex}_{os.path.basename(file)}"
        path = os.path.join("user_midi", filename)
        shutil.copy(file, path)
        midi_paths.append(path)

    train_model_on_midis(midi_paths)
    model_trained = True
    return gr.update(value="✅ Training complete! You can now generate music 🎵"), True

def generate_music_wrapper(num_notes):
    if not model_trained:
        return None, None
    wav_path = generate_music(num_notes=num_notes)
    return wav_path, wav_path  # same file for playback and download

with gr.Blocks(css="""
    #footer {
        text-align: center;
        margin-top: 2em;
        font-size: 0.9em;
        color: gray;
    }
    #footer a {
        color: #00bfff;
        text-decoration: none;
    }
    #footer a:hover {
        text-decoration: underline;
    }
""") as demo:
    gr.Markdown("## 🎵 AI Music Generator")
    gr.Markdown("Upload **3 or more** MIDI files, train your own model, and generate music!")

    with gr.Group(elem_id="upload-section"):
        midi_files = gr.File(
            label="🎼 Upload MIDI Files",
            file_types=[".mid", ".midi"],
            file_count="multiple"
        )
        train_btn = gr.Button("🚀 Train Model")
        training_status = gr.Textbox(label="Training Status", interactive=False)

    train_btn.click(fn=train_model, inputs=midi_files, outputs=[training_status, gr.State()])

    gr.Markdown("### 🎚️ Number of Notes to Generate")
    note_slider = gr.Slider(minimum=50, maximum=500, value=100, step=1, label="Number of Notes")

    generate_btn = gr.Button("🎶 Generate Music")
    audio_output = gr.Audio(label="🔊 Listen to Generated Music", type="filepath", autoplay=True)
    download_file = gr.File(label="⬇️ Download WAV File")

    generate_btn.click(fn=generate_music_wrapper, inputs=note_slider, outputs=[audio_output, download_file])

    gr.Markdown('<div id="footer">Made with ❤️ by <a href="https://github.com/shivamcy" target="_blank">shivamcy</a></div>', elem_id="footer")

if __name__ == "__main__":
    demo.launch()
