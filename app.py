
import streamlit as st
from pathlib import Path

from musicgenerator import (
    train_model,
    load_trained_model,
    generate_events,
    events_to_midi,
    midi_to_wav,
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    MODEL_PATH,
    VOCAB_PATH,
    DEVICE
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="MelodyMind AI",
    page_icon="🎵",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101a35, #172d50);
    color: white;
}
h1, h2, h3, p, label, span {
    color: white;
}
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: bold;
    color: #69c7ff;
}
.subtitle {
    text-align: center;
    color: #d0dbea;
    font-size: 18px;
}
div.stButton > button {
    background-color: #1688d4;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0865a8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🎵 MelodyMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An AI-Powered Music Generation System Using LSTM Neural Networks</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- STATUS ----------------

midi_files = list(DATA_DIR.glob("*.mid")) + list(DATA_DIR.glob("*.midi"))

model_exists = MODEL_PATH.exists() and VOCAB_PATH.exists()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MIDI Dataset Files", len(midi_files))

with col2:
    st.metric(
        "AI Model",
        "Trained" if model_exists else "Not Trained"
    )

with col3:
    st.metric("Processing Device", str(DEVICE).upper())

st.divider()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🎼 MelodyMind AI")
    st.write("AI Music Generation Dashboard")
    st.divider()

    st.subheader("Project Information")
    st.write("**Algorithm:** LSTM")
    st.write("**Framework:** PyTorch")
    st.write("**Input:** MIDI files")
    st.write("**Output:** MIDI and WAV")
    st.divider()

    st.info(
        "Add your MIDI training files inside "
        "the data/midi folder."
    )

# ---------------- TRAINING SECTION ----------------

st.header("🧠 Train Your AI Model")

st.write(
    "Train the LSTM neural network using your MIDI music dataset."
)

with st.form("training_form"):

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input(
            "Training Epochs",
            min_value=1,
            max_value=100,
            value=10
        )

    with col2:
        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128],
            index=2
        )

    train_button = st.form_submit_button(
        "🚀 Start Model Training",
        use_container_width=True
    )

if train_button:

    if len(midi_files) == 0:
        st.error(
            "No MIDI files found! Add .mid or .midi files "
            "inside data/midi first."
        )

    else:
        progress_message = st.empty()

        try:
            with st.spinner("Training your AI music model..."):
                progress_message.info(
                    "Please wait. Training may take some time."
                )

                result = train_model(
                    epochs=int(epochs),
                    batch_size=int(batch_size)
                )

            st.success("AI model trained successfully!")

            st.write(
                f"**Training sequences:** "
                f"{result['training_sequences']}"
            )

            st.write(
                f"**Vocabulary size:** "
                f"{result['vocabulary_size']}"
            )

            st.write(
                f"**Model saved:** {result['model_path']}"
            )

            st.rerun()

        except Exception as error:
            st.error(f"Training failed: {error}")

st.divider()

# ---------------- MUSIC GENERATION SECTION ----------------

st.header("🎹 Generate New Music")

st.write(
    "Use your trained LSTM model to generate an original "
    "melody based on learned musical patterns."
)

with st.form("generation_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        number_of_notes = st.slider(
            "Number of Notes",
            min_value=20,
            max_value=300,
            value=100
        )

    with col2:
        temperature = st.slider(
            "Creativity / Temperature",
            min_value=0.2,
            max_value=1.5,
            value=0.9,
            step=0.1
        )

    with col3:
        bpm = st.slider(
            "Music Speed (BPM)",
            min_value=50,
            max_value=180,
            value=100
        )

    generate_button = st.form_submit_button(
        "🎶 Generate Music",
        use_container_width=True
    )

if generate_button:

    if not model_exists:
        st.warning(
            "Please train the AI model before generating music."
        )

    else:
        try:
            with st.spinner("Composing your music..."):

                model, vocabulary = load_trained_model()

                generated_events = generate_events(
                    model,
                    vocabulary,
                    number_of_notes=number_of_notes,
                    temperature=temperature
                )

                midi_path = OUTPUT_DIR / "generated_melody.mid"

                events_to_midi(
                    generated_events,
                    midi_path,
                    bpm=bpm
                )

                wav_path = OUTPUT_DIR / "generated_melody.wav"

                midi_to_wav(
                    midi_path,
                    wav_path
                )

            st.success("Your AI-generated music is ready!")

            st.subheader("🎧 Listen to Your Music")

            st.audio(
                str(wav_path),
                format="audio/wav"
            )

            st.subheader("📥 Download Your Music")

            col1, col2 = st.columns(2)

            with col1:
                with open(midi_path, "rb") as file:
                    st.download_button(
                        "Download MIDI",
                        data=file,
                        file_name="generated_melody.mid",
                        mime="audio/midi",
                        use_container_width=True
                    )

            with col2:
                with open(wav_path, "rb") as file:
                    st.download_button(
                        "Download WAV",
                        data=file,
                        file_name="generated_melody.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

            st.write(f"Generated notes: {len(generated_events)}")

        except Exception as error:
            st.error(f"Music generation failed: {error}")

st.divider()

# ---------------- FOOTER ----------------

st.markdown(
    """
    <div style="text-align:center; color:#a9bdd8; padding:20px;">
        MelodyMind AI | AI Music Generation Project<br>
        Developed using Python, PyTorch and LSTM
    </div>
    """,
    unsafe_allow_html=True
)


