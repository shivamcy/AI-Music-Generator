import gradio as gr
import os
from train_model import train_model_on_midis
from generate_music import generate_music_from_trained_model
import shutil

# Ensure necessary folders exist
os.makedirs("user_midi", exist_ok=True)
os.makedirs("static/generated_music", exist_ok=True)

# State flag
model_trained = False

def train_model(midi_files):
    global model_trained
    if len(midi_files) < 3:
        return gr.update(value="❌ Please upload at least 3 MIDI files."), False
    # Save uploaded files
    midi_paths = []
    for file in midi_files:
        path = os.path.join("user_midi", file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        midi_paths.append(path)
    # Train the model
    train_model_on_midis(midi_paths)
    model_trained = True
    return gr.update(value="✅ Training complete! You can now generate music 🎵"), True

def generate_music_wrapper(num_notes):
    if not model_trained:
        return None
    output_path = generate_music_from_trained_model(num_notes)

    final_output = "static/generated_music/output.mid"
    shutil.move(output_path, final_output)
    return final_output

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
    gr.Markdown("Upload 3+ MIDI files, train a model, and generate your own AI music!\nRuns in-browser. Files are private.")

    midi_files = gr.File(label="🎼 Upload MIDI Files", file_types=[".mid", ".midi"], file_count="multiple")

    train_btn = gr.Button("🚀 Train Model")
    training_status = gr.Textbox(label="Training Status", interactive=False)

    train_btn.click(fn=train_model, inputs=midi_files, outputs=[training_status, gr.State()])

    gr.Markdown("### 🎚️ Number of Notes")
    note_slider = gr.Slider(minimum=50, maximum=500, value=100, step=1, label="Number of Notes")

    generate_btn = gr.Button("🎶 Generate Music")
    output_file = gr.File(label="🎼 Download Your Generated MIDI File")

    generate_btn.click(fn=generate_music_wrapper, inputs=note_slider, outputs=output_file)

    gr.Markdown('<div id="footer">Developed by <a href="https://github.com/shivamcy" target="_blank">shivamcy</a></div>', elem_id="footer")

if __name__ == "__main__":
    demo.launch()
