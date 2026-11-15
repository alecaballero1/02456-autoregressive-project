import torch
import csv
from pathlib import Path
from ..models.lstm_pytorch import LSTMRNN
from ..data.split_data import train_data
from ..constants.lstm_constants import CSV_FILENAME, MODEL_NAME, MODEL_NB

# Ensure script is run on GPU rather than CPU
device = torch.device("cuda")

# current location is src/protgen/train_scripts, .parents[2] will provide path to src/
src_path = Path(__file__).resolve().parents[2]
model_path = src_path / "protgen" / "generated" / "lstm" / str(MODEL_NB) / MODEL_NAME
model_state = torch.load(model_path, map_location=device, weights_only=False)
sample_path = src_path.parent / "results" / "sequences" / str(MODEL_NB)
sample_path.mkdir(parents=True, exist_ok=True)
sample_path = sample_path / CSV_FILENAME

# Recreate model using configuration of the saved state
net = LSTMRNN(
    n_in=model_state["vocabulary_size"],
    n_hid=model_state["hidden_size"],
    nb_layers=1,
    bias=True,
    vocab_size=model_state["vocabulary_size"]
).to(device)

# Load the saved training weights
net.load_state_dict(model_state["model"])
net.eval()

# Sample
with torch.no_grad():
    samples = net.sample(batch_size=100,num_steps=500,temperature=0.8)

sequences = [train_data.decode(sample.cpu().tolist()) for sample in samples]

# write samples to csv file
with open(sample_path, "w", newline = "") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["sequence"])
    writer.writeheader()
    writer.writerows({"sequence": seq} for seq in sequences)

print(f"Generated {len(sequences)} sequences")