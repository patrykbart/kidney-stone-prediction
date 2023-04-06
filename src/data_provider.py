import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler


class KidneyStoneDataset(torch.utils.data.Dataset):
    def __init__(self, path, scaler=None):
        self.path = path

        df = pd.read_csv(self.path)

        if "id" in df.columns:
            df.drop(["id"], axis=1, inplace=True)

        if "gravity" in df.columns:
            df.drop(["gravity"], axis=1, inplace=True)

        if "target" in df.columns:
            self.X = torch.tensor(df.drop(["target"], axis=1).values, dtype=torch.float32)
            self.y = torch.tensor(df["target"].values, dtype=torch.float32)
        else:
            self.X = torch.tensor(df.values, dtype=torch.float32)
            self.y = None

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
        else:
            self.scaler = scaler

        self.X = torch.tensor(self.scaler.transform(self.X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx].unsqueeze(0)
        else:
            return self.X[idx]
