import torch
import faiss
import numpy as np
import os
import yaml
from src.model import TwoTowerModel

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_item_index(config_path="config.yaml"):
    cfg = load_config(config_path)
    
    print("Loading model for indexing...")
    device = torch.device("cpu") # CPU is sufficient for indexing 5k items
    
    # 1. Load Model
    model = TwoTowerModel(
        num_users=cfg['model']['num_users'],
        num_items=cfg['model']['num_items'],
        embedding_dim=cfg['model']['embedding_dim'],
        hidden_dim=cfg['model']['hidden_dim']
    )
    
    model_path = os.path.join(cfg['training']['save_dir'], "two_tower_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first!")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Generate Embeddings for ALL items
    print("Generating item embeddings...")
    all_item_ids = torch.arange(cfg['model']['num_items'], dtype=torch.long)
    
    with torch.no_grad():
        item_emb = model.item_embedding(all_item_ids)
        item_vectors = model.item_fc(item_emb)
        # Normalize for Cosine Similarity
        item_vectors = torch.nn.functional.normalize(item_vectors, p=2, dim=1)
        
    item_vectors_np = item_vectors.cpu().numpy()

    # 3. Build FAISS Index
    print("Building FAISS index...")
    # IndexFlatIP = Inner Product (which equals Cosine Sim if vectors are normalized)
    index = faiss.IndexFlatIP(cfg['model']['hidden_dim'])
    index.add(item_vectors_np)
    
    # 4. Save Index
    os.makedirs("model_artifacts", exist_ok=True)
    index_path = "model_artifacts/item_index.faiss"
    faiss.write_index(index, index_path)
    
    print(f"Index built with {index.ntotal} items and saved to {index_path}")

if __name__ == "__main__":
    build_item_index()