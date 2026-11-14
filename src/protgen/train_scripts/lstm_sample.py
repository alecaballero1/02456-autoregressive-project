import torch
import csv
from pathlib import Path
from ..models.lstm_pytorch import LSTMRNN
from ..data.split_data import train_data
from ..constants.lstm_constants import BOS_IDX, PAD_IDX, EOS_IDX, CSV_FILENAME, MODEL_NAME

# Ensure script is run on GPU rather than CPU
device = torch.device("cuda")

# current location is src/protgen/train_scripts, .parents[2] will provide path to src/
src_path = Path(__file__).resolve().parents[2]
model_path = src_path / "protgen" / "generated" / "lstm" / MODEL_NAME
model_state = torch.load(model_path, map_location=device, weights_only=False)

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
with open(src_path / "results" / "sequences" / CSV_FILENAME, "w", newline = "") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["sequence"])
    writer.writeheader()
    writer.writerows({"sequence": seq} for seq in sequences)

print(f"Generated {len(sequences)} sequences.")