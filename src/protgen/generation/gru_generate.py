import csv
from pathlib import Path

import torch

from protgen.helpers.gru_constants import (
    CHECKPOINT_DIR,
    SEQUENCE_DIR,
    TEMPERATURE,
    SEED,
)
from protgen.models.gru import Gru

class GenerateGru:
    def __init__(self, pt_file_name):
        self.pt_file_path = CHECKPOINT_DIR / pt_file_name
        self.output_path = SEQUENCE_DIR / f"{self.pt_file_path.stem}.csv"
    
    def generate(self):
        num_sequences = 256
        max_length = 127

        torch.manual_seed(SEED)

        checkpoint = torch.load(
            self.pt_file_path,
            map_location="cpu",
            weights_only=True,
        )

        vocab = checkpoint["vocab"]
        id_to_token = {
            token_id: token
            for token, token_id in vocab.items()
        }

        model = Gru(
            vocab_size=len(vocab),
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        rows = []

        with torch.no_grad():
            for index in range(num_sequences):
                token = torch.tensor([[vocab["<BOS>"]]])
                hidden = None
                residues = []
                ended_with_eos = False

                for position in range(max_length + 1):
                    logits, hidden = model(token, hidden)
                    scores = logits[0, -1] / TEMPERATURE

                    scores[vocab["<PAD>"]] = -torch.inf
                    scores[vocab["<BOS>"]] = -torch.inf

                    probabilities = torch.softmax(scores, dim=-1)
                    token = torch.multinomial(
                        probabilities, num_samples=1
                    ).reshape(1, 1)

                    token_id = token.item()

                    if token_id == vocab["<EOS>"]:
                        ended_with_eos = True
                        break

                    if position == max_length:
                        break

                    residues.append(id_to_token[token_id])

                rows.append([
                    f"gru_{index}",
                    "".join(residues),
                    ended_with_eos,
                ])

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "sequence", "ended_with_eos"])
            writer.writerows(rows)

        print(f"Saved {num_sequences} sequences to {self.output_path}")


if __name__ == "__main__":
    filename = input(f"Provide checkpoint file from {CHECKPOINT_DIR} (without .pt):\n") + ".pt"
    GenerateGru(filename).generate()
