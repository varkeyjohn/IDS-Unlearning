import torch
from torch.utils.data import Dataset
import pandas as pd

class IDSDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = df.drop(columns=["label"]).values.astype('float32')
        self.y = df["label"].astype('category').cat.codes.values  # Convert labels to ints

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
