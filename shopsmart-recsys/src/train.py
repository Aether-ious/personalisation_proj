import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os

# Import our custom modules
from src.model import TwoTowerModel
from src.dataset import ShopSmartDataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train():
    # 1. Load Configuration
    cfg = load_config()
    
    # 2. Setup Device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 3. Prepare Data
    print("Loading Datasets...")
    train_dataset = ShopSmartDataset(cfg['data']['train_path'])
    
    # DataLoader handles batching and shuffling
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=True,
        num_workers=2 # Parallel data loading
    )

    # 4. Initialize Model
    model = TwoTowerModel(
        num_users=cfg['model']['num_users'],
        num_items=cfg['model']['num_items'],
        embedding_dim=cfg['model']['embedding_dim'],
        hidden_dim=cfg['model']['hidden_dim']
    ).to(device)

    # 5. Loss and Optimizer
    # BCEWithLogitsLoss is standard for binary classification (Click vs No-Click)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    # 6. Training Loop
    epochs = cfg['training']['epochs']
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch data to GPU/CPU
            user_ids = batch['user_id'].to(device)
            item_ids = batch['item_id'].to(device)
            targets = batch['target'].to(device)

            # --- A. Zero Gradients ---
            # Clear previous gradients so they don't accumulate
            optimizer.zero_grad()

            # --- B. Forward Pass ---
            # Get predictions from the model
            predictions = model(user_ids, item_ids)

            # --- C. Calculate Loss ---
            loss = criterion(predictions, targets)

            # --- D. Backward Pass ---
            # Compute gradients based on loss
            loss.backward()

            # --- E. Optimization Step ---
            # Update model weights
            optimizer.step()

            total_loss += loss.item()

            # Logging every 10 batches
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # 7. Save the Model
    if not os.path.exists(cfg['training']['save_dir']):
        os.makedirs(cfg['training']['save_dir'])
        
    save_path = os.path.join(cfg['training']['save_dir'], "two_tower_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()