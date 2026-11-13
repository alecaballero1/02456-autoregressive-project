import torch
from torch import nn
from torch.utils.data import DataLoader

from protgen.helpers.gru_constants import CHECKPOINT_DIR, BATCH_SIZE
from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.helpers.gru_epochs import evaluate
from protgen.models.gru import Gru


class TestGru:
    def __init__(self, pt_file_name):
        self.pt_file_path = CHECKPOINT_DIR / pt_file_name
        self.test_loss_path = CHECKPOINT_DIR / f"{self.pt_file_path.stem}_test_loss.txt"

    def test(self):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            self.pt_file_path,
            map_location=device,
            weights_only=True,
        )

        model = Gru(
            vocab_size=len(checkpoint["vocab"]),
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint.get("num_layers", 1),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])

        test_data = ProteinDataset("data/test.csv")
        test_loader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        loss_function = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )

        test_loss = evaluate(
            model,
            test_loader,
            loss_function,
            device,
        )

        print(f"Model: {self.pt_file_path.name}")
        print(f"Test loss: {test_loss:.4f}")
        
        return test_loss


if __name__ == "__main__":
    filename = input(f"Provide checkpoint file to test from {CHECKPOINT_DIR} (without .pt):\n") + ".pt"
    TestGru(filename).test()
