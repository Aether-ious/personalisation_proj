import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class ShopSmartDataset(Dataset):
    """
    Custom Dataset for Two-Tower RecSys.
    Expected columns: [user_id, item_id, target]
    target: 1 (click/purchase) or 0 (no interaction/negative sample)
    """
    def __init__(self, csv_path):
        # Load data using Pandas (for speed)
        self.data = pd.read_csv(csv_path)
        
        # Convert to numpy for faster indexing during training
        self.users = self.data['user_id'].values
        self.items = self.data['item_id'].values
        self.targets = self.data['target'].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # This method is called by the DataLoader to fetch a batch
        user = self.users[idx]
        item = self.items[idx]
        target = self.targets[idx]
        
        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "item_id": torch.tensor(item, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float32)
        }