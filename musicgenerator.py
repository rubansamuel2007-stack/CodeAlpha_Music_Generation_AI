import json
import random
from pathlib import Path

import numpy as np
import pretty_midi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "midi"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = MODEL_DIR / "melodymind_lstm.pt"
VOCAB_PATH = MODEL_DIR / "vocabulary.json"

SEQUENCE_LENGTH = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2

DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 64

DURATION_BINS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Reproducible initial training behavior.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# MIDI DATA PROCESSING
# ==================================================

def nearest_duration(duration):
    """
    Map a note duration to the closest supported
    duration value, measured in beats.
    """

    duration = max(float(duration), 0.1)

    return min(
        DURATION_BINS,
        key=lambda value: abs(value - duration)
    )


def midi_to_events(file_path):
    """
    Read a MIDI file and extract note pitch-duration events.

    Each event is represented as:
        pitch:duration

    Example:
        60:1.0
    """

    midi = pretty_midi.PrettyMIDI(str(file_path))

    tempo_changes = midi.get_tempo_changes()

    if len(tempo_changes[1]) > 0:
        tempo = float(np.median(tempo_changes[1]))
    else:
        tempo = 120.0

    tempo = max(tempo, 1.0)

    seconds_per_beat = 60.0 / tempo

    notes = []

    for instrument in midi.instruments:

        for note in instrument.notes:

            if note.end <= note.start:
                continue

            # Convert duration from seconds to approximate beats.
            duration_seconds = note.end - note.start

            duration_beats = (
                duration_seconds / seconds_per_beat
            )

            duration_beats = nearest_duration(
                duration_beats
            )

            notes.append(
                (
                    float(note.start),
                    int(note.pitch),
                    float(duration_beats)
                )
            )

    # Process events in chronological order.
    notes.sort(key=lambda item: item[0])

    events = [
        f"{pitch}:{duration}"
        for _, pitch, duration in notes
    ]

    return events


def collect_training_events():
    """
    Read every MIDI file in the dataset folder
    and combine its musical events.
    """

    midi_files = sorted(
        list(DATA_DIR.glob("*.mid"))
        + list(DATA_DIR.glob("*.midi"))
    )

    if not midi_files:
        raise FileNotFoundError(
            "No MIDI files were found. "
            "Add .mid or .midi files to data/midi/."
        )

    all_events = []

    successful_files = 0

    for midi_file in midi_files:

        try:
            events = midi_to_events(midi_file)

            if len(events) >= SEQUENCE_LENGTH + 1:

                all_events.extend(events)

                # Separator between different songs.
                all_events.append("<SEP>")

                successful_files += 1

                print(
                    f"Loaded {midi_file.name}: "
                    f"{len(events)} events"
                )

            else:
                print(
                    f"Skipped {midi_file.name}: "
                    "not enough note events."
                )

        except Exception as error:

            print(
                f"Could not process {midi_file.name}: {error}"
            )

    if successful_files == 0:

        raise ValueError(
            "No usable MIDI files were found. "
            "Try adding longer, valid MIDI files."
        )

    print(f"\nSuccessfully loaded {successful_files} files.")

    return all_events


# ==================================================
# VOCABULARY AND SEQUENCES
# ==================================================

def create_vocabulary(events):
    """
    Assign an integer ID to every musical event.
    """

    unique_events = sorted(
        set(events) - {"<SEP>"}
    )

    # Reserve 0 for the separator.
    vocabulary = {"<SEP>": 0}

    for event in unique_events:
        vocabulary[event] = len(vocabulary)

    return vocabulary


def create_training_sequences(events, vocabulary):
    """
    Build overlapping sequences for next-event prediction.
    """

    encoded = [
        vocabulary[event]
        for event in events
    ]

    x_data = []
    y_data = []

    for index in range(
        len(encoded) - SEQUENCE_LENGTH
    ):

        sequence = encoded[
            index:index + SEQUENCE_LENGTH
        ]

        target = encoded[
            index + SEQUENCE_LENGTH
        ]

        x_data.append(sequence)
        y_data.append(target)

    return (
        np.array(x_data, dtype=np.int64),
        np.array(y_data, dtype=np.int64)
    )


class MusicDataset(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.tensor(
            x_data,
            dtype=torch.long
        )

        self.y_data = torch.tensor(
            y_data,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return (
            self.x_data[index],
            self.y_data[index]
        )


# ==================================================
# LSTM MODEL
# ==================================================

class MusicLSTM(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_size=64,
        hidden_size=128,
        num_layers=2
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_size
        )

        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(0.2)

        self.output_layer = nn.Linear(
            hidden_size,
            vocabulary_size
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, _ = self.lstm(embedded)

        last_output = output[:, -1, :]

        last_output = self.dropout(last_output)

        logits = self.output_layer(last_output)

        return logits


# ==================================================
# TRAIN THE MODEL
# ==================================================

def train_model(
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE
):
    """
    Process the MIDI dataset, train the LSTM,
    and save the model and vocabulary.
    """

    events = collect_training_events()

    vocabulary = create_vocabulary(events)

    x_data, y_data = create_training_sequences(
        events,
        vocabulary
    )

    if len(x_data) < 10:

        raise ValueError(
            "Not enough training sequences. "
            "Add more MIDI data or use longer songs."
        )

    print(f"\nVocabulary size: {len(vocabulary)}")
    print(f"Training sequences: {len(x_data)}")
    print(f"Device: {DEVICE}")

    dataset = MusicDataset(x_data, y_data)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    model = MusicLSTM(
        vocabulary_size=len(vocabulary),
        embedding_size=64,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        for inputs, targets in loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(inputs)

            loss = loss_function(
                predictions,
                targets
            )

            loss.backward()

            # Prevent excessively large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / max(
            len(loader),
            1
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    # Save model weights and configuration.
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocabulary_size": len(vocabulary),
        "embedding_size": 64,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "sequence_length": SEQUENCE_LENGTH
    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocabulary,
            file,
            indent=2
        )

    print("\nTraining completed!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vocabulary saved to: {VOCAB_PATH}")

    return {
        "epochs": epochs,
        "training_sequences": len(x_data),
        "vocabulary_size": len(vocabulary),
        "model_path": str(MODEL_PATH)
    }


# ==================================================
# LOAD A TRAINED MODEL
# ==================================================

def load_trained_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "The trained model was not found. "
            "Train the model before generating music."
        )

    if not VOCAB_PATH.exists():

        raise FileNotFoundError(
            "The vocabulary file is missing. "
            "Train the model again."
        )

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(file)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = MusicLSTM(
        vocabulary_size=checkpoint["vocabulary_size"],
        embedding_size=checkpoint["embedding_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, vocabulary


# ==================================================
# GENERATE MUSIC
# ==================================================

def sample_next_token(logits, temperature=0.9):
    """
    Select the next event using temperature sampling.

    Lower temperatures generally produce more predictable
    sequences. Higher temperatures allow more variation.
    """

    temperature = max(
        float(temperature),
        0.1
    )

    logits = logits / temperature

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    # Avoid selecting the separator token as a musical note.
    probabilities[0] = 0

    probability_sum = probabilities.sum()

    if probability_sum <= 0:

        probabilities = torch.ones_like(
            probabilities
        )

        probabilities[0] = 0

        probabilities = probabilities / probabilities.sum()

    else:

        probabilities = probabilities / probability_sum

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    return int(next_token.item())


def generate_events(
    model,
    vocabulary,
    number_of_notes=100,
    temperature=0.9
):
    """
    Generate a sequence of musical events.
    """

    reverse_vocabulary = {
        index: event
        for event, index in vocabulary.items()
    }

    available_tokens = [
        token
        for token in vocabulary.values()
        if token != vocabulary["<SEP>"]
    ]

    if len(available_tokens) == 0:

        raise ValueError(
            "The vocabulary contains no musical events."
        )

    # Start with a random learned musical event.
    starting_token = random.choice(
        available_tokens
    )

    sequence = [
        starting_token
    ] * SEQUENCE_LENGTH

    generated_events = []

    model.eval()

    with torch.no_grad():

        for _ in range(number_of_notes):

            input_sequence = torch.tensor(
                [sequence[-SEQUENCE_LENGTH:]],
                dtype=torch.long,
                device=DEVICE
            )

            logits = model(
                input_sequence
            )[0]

            next_token = sample_next_token(
                logits,
                temperature
            )

            event = reverse_vocabulary.get(
                next_token
            )

            if event is None or event == "<SEP>":
                continue

            generated_events.append(event)

            sequence.append(next_token)

    if not generated_events:

        raise RuntimeError(
            "The model did not generate musical events."
        )

    return generated_events


# ==================================================
# CONVERT EVENTS TO MIDI
# ==================================================

def events_to_midi(
    events,
    output_path,
    bpm=100
):
    """
    Convert pitch-duration events into a MIDI melody.
    """

    midi = pretty_midi.PrettyMIDI(
        initial_tempo=bpm
    )

    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(
            "Acoustic Grand Piano"
        )
    )

    seconds_per_beat = 60.0 / bpm

    current_time = 0.0

    for event in events:

        try:

            pitch_text, duration_text = event.split(":")

            pitch = int(pitch_text)

            duration_beats = float(duration_text)

        except (ValueError, TypeError):
            continue

        # Restrict pitch to the valid MIDI range.
        if not 0 <= pitch <= 127:
            continue

        duration_seconds = (
            duration_beats * seconds_per_beat
        )

        note = pretty_midi.Note(
            velocity=90,
            pitch=pitch,
            start=current_time,
            end=current_time + duration_seconds
        )

        instrument.notes.append(note)

        # Sequential melody with no overlapping notes.
        current_time += duration_seconds

    if not instrument.notes:

        raise RuntimeError(
            "No valid notes were produced."
        )

    midi.instruments.append(instrument)

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    midi.write(str(output_path))

    return str(output_path)


# ==================================================
# CONVERT MIDI TO WAV
# ==================================================

def midi_to_wav(
    midi_path,
    wav_path
):
    """
    Create a WAV preview using the synthesizer
    available through pretty_midi.

    The fallback synthesizer may sound basic.
    """

    from scipy.io import wavfile

    midi = pretty_midi.PrettyMIDI(
        str(midi_path)
    )

    sample_rate = 22050

    audio = midi.synthesize(
        fs=sample_rate
    )

    if audio is None or len(audio) == 0:

        raise RuntimeError(
            "Audio synthesis produced an empty file."
        )

    # Normalize audio safely.
    peak = float(np.max(np.abs(audio)))

    if peak > 0:

        audio = audio / peak

    audio = (
        audio * 32767
    ).astype(np.int16)

    wav_path = Path(wav_path)

    wav_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    wavfile.write(
        str(wav_path),
        sample_rate,
        audio
    )

    return str(wav_path)


# ==================================================
# COMMAND-LINE ENTRY POINT
# ==================================================

if __name__ == "__main__":

    print("\n🎹 MelodyMind AI Music Generator")
    print("================================")

    print("\n1. Train model")
    print("2. Generate music")

    choice = input(
        "\nSelect an option (1 or 2): "
    ).strip()

    if choice == "1":

        epochs_text = input(
            "Number of training epochs [30]: "
        ).strip()

        epochs = (
            int(epochs_text)
            if epochs_text
            else DEFAULT_EPOCHS
        )

        train_model(
            epochs=epochs
        )

    elif choice == "2":

        model, vocabulary = load_trained_model()

        generated = generate_events(
            model,
            vocabulary,
            number_of_notes=100,
            temperature=0.9
        )

        midi_path = OUTPUT_DIR / "generated_melody.mid"

        events_to_midi(
            generated,
            midi_path,
            bpm=100
        )

        wav_path = OUTPUT_DIR / "generated_melody.wav"

        midi_to_wav(
            midi_path,
            wav_path
        )

        print("\nMusic generated successfully!")
        print(f"MIDI: {midi_path}")
        print(f"WAV: {wav_path}")

    else:

        print("Invalid option. Please select 1 or 2.")