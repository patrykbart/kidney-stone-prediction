import torch
import pandas as pd
import pytorch_lightning as pl

from src.model import BinaryClassifier
from src.data_provider import KidneyStoneDataset
from src.utils import get_config_yaml, visible_print

if __name__ == "__main__":
    config = get_config_yaml()

    visible_print("Loading data")
    train_dataset = KidneyStoneDataset(config["data"]["train_path"])
    valid_dataset = KidneyStoneDataset(config["data"]["valid_path"], scaler=train_dataset.scaler)
    test_dataset = KidneyStoneDataset(config["data"]["test_path"], scaler=train_dataset.scaler)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    visible_print("Initializing trainer")
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="valid_loss",
            dirpath=config["training"]["checkpoint_dir"],
            filename="best-checkpoint",
            save_top_k=1,
            mode="min",
        ),
        # pl.callbacks.EarlyStopping(
        #     monitor="valid_loss",
        #     patience=config["training"]["early_stopping_patience"],
        #     mode="min",
        # ),
    ]

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        enable_model_summary=True,
        logger=False,
        callbacks=callbacks,
    )

    visible_print("Model summary")
    model = BinaryClassifier(config)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    visible_print("Testing")
    model = BinaryClassifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, config=config)
    trainer.test(model=model, dataloaders=test_loader)

    visible_print("Save test predictions")
    df = pd.read_csv(config["data"]["test_path"])
    df["target"] = model.test_preds
    df.drop([col for col in df.columns if col not in ["id", "target"]], axis=1, inplace=True)

    df.to_csv(config["data"]["test_predictions_path"], index=False)
    print(f"Test predictions saved to {config['data']['test_predictions_path']}")
