import csv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime

from protgen.helpers.gru_constants import (
    CHECKPOINT_DIR,
    EPOCHS,
    BATCH_SIZE,
    GRU_HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_GRU_LAYERS,
    GRADIENT_CLIP_NORM,
    WEIGHT_DECAY,
    SEED,
)
from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models.gru import Gru
from protgen.helpers.gru_epochs import train_epoch, evaluate


class TrainGru:
    def __init__(self):
        self.seed = SEED

        self.time_stamp = datetime.now().strftime("%y%m%d%H%M")
        rate_label = f"{LEARNING_RATE:.10f}".removeprefix("0.").rstrip("0")
        self.run_name = (
            f"gru_h{GRU_HIDDEN_SIZE}_l{NUM_GRU_LAYERS}_"
            f"r{rate_label}_e{EPOCHS}_{self.time_stamp}"
        )
        self.checkpoint_path = CHECKPOINT_DIR / f"{self.run_name}.pt"
        self.history_path = CHECKPOINT_DIR / f"{self.run_name}_history.csv"

    def train(self):

        torch.manual_seed(self.seed)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        train_data = ProteinDataset("data/train.csv")
        val_data = ProteinDataset("data/val.csv")

        train_loader = DataLoader(
            train_data,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = Gru(
            vocab_size=train_data.vocab_size,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=NUM_GRU_LAYERS,
        ).to(device)

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        loss_function = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")

        print(f"Training on {device}", flush=True)

        with self.history_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss"])

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_epoch(
                    model,
                    train_loader,
                    loss_function,
                    optimizer,
                    device,
                )

                val_loss = evaluate(
                    model,
                    val_loader,
                    loss_function,
                    device,
                )

                writer.writerow([epoch, train_loss, val_loss])
                handle.flush()

                print(
                    f"Epoch {epoch}: "
                    f"train={train_loss:.4f}, val={val_loss:.4f}",
                    flush=True,
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                    torch.save({
                        "model_state": model.state_dict(),
                        "vocab": train_data.vocab,
                        "hidden_size": model.rnn_layer.hidden_size,
                        "num_layers": model.rnn_layer.num_layers,
                        "dropout": model.rnn_layer.dropout,
                        "gradient_clip_norm": GRADIENT_CLIP_NORM,
                        "best_epoch": epoch,
                        "total_epochs": EPOCHS,
                        "val_loss": val_loss,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "batch_size": BATCH_SIZE,
                        "optimizer": "RMSprop",
                        "seed": self.seed,
                    }, self.checkpoint_path)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(checkpoint["model_state"])

        print(
            f"Best epoch: {checkpoint['best_epoch']} "
            f"with validation loss: {checkpoint['val_loss']:.4f}"
        )

        return model


if __name__ == "__main__":
    model = TrainGru().train()
