import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from src.utils import visible_print


class BinaryClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]

        # Define model architecture
        self.fc_in = nn.Linear(self.model_config["input_dim"], self.model_config["hidden_dim"])
        for i in range(self.model_config["num_hidden_layers"]):
            setattr(self, f"fc_{i}", nn.Linear(self.model_config["hidden_dim"], self.model_config["hidden_dim"]))
        self.fc_out = nn.Linear(self.model_config["hidden_dim"], 1)

        # Define loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Define metrics
        self.acc = torchmetrics.Accuracy(task="binary")

        # Store test predictions
        self.test_preds = []

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config["weight_decay"],
        )

    def on_train_start(self):
        visible_print("Training")

    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(x)
        for i in range(self.model_config["num_hidden_layers"]):
            x = getattr(self, f"fc_{i}")(x)
            x = F.relu(x)
        x = self.fc_out(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        acc = self.acc(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        acc = self.acc(y_hat, y)

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)

        self.test_preds.extend(y_hat.detach().cpu().numpy().tolist())

    def on_test_end(self):
        self.test_preds = torch.sigmoid(torch.tensor(self.test_preds).squeeze(-1)).numpy().tolist()
