import argparse
import csv

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from protgen.helpers.gru_constants import (
    CHECKPOINT_DIR,
    EPOCHS,
    BATCH_SIZE,
    GRU_SIZE,
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
    def __init__(self, **args):
        self.seed = SEED
        self.epochs = args.get("epochs", EPOCHS)
        self.num_gru_layers = args.get("num_gru_layers", NUM_GRU_LAYERS)
        self.gru_size = args.get("gru_size", GRU_SIZE)
        self.learning_rate = args.get("learning_rate", LEARNING_RATE)
        self.batch_size = args.get("batch_size", BATCH_SIZE)
        self.weight_decay = args.get("weight_decay", WEIGHT_DECAY)

        rate_label = f"{self.learning_rate:.10f}".removeprefix("0.").rstrip("0")
        self.run_name = (
            f"gru_h{self.gru_size}_l{self.num_gru_layers}_"
            f"r{rate_label}_e{self.epochs}_"
            f"b{self.batch_size}_d{self.weight_decay}"
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
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
        )

        model = Gru(
            vocab_size=train_data.vocab_size,
            hidden_size=self.gru_size,
            num_layers=self.num_gru_layers,
        ).to(device)

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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

            for epoch in range(1, self.epochs + 1):
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
                        "total_epochs": self.epochs,
                        "val_loss": val_loss,
                        "learning_rate": self.learning_rate,
                        "weight_decay": self.weight_decay,
                        "batch_size": self.batch_size,
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
    parser = argparse.ArgumentParser(description="Train a GRU protein sequence model.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-gru-layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gru-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    args = parser.parse_args()

    model = TrainGru(**vars(args)).train()
