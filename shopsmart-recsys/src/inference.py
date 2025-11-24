import torch
import faiss
import yaml
import os
import numpy as np
from src.model import TwoTowerModel

class RecSysService:
    def __init__(self, config_path="config.yaml"):
        self.cfg = self._load_config(config_path)
        self.device = torch.device("cpu") 
        
        self._load_model()
        self._load_index()
        
    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_model(self):
        self.model = TwoTowerModel(
            num_users=self.cfg['model']['num_users'],
            num_items=self.cfg['model']['num_items'],
            embedding_dim=self.cfg['model']['embedding_dim'],
            hidden_dim=self.cfg['model']['hidden_dim']
        )
        model_path = os.path.join(self.cfg['training']['save_dir'], "two_tower_model.pth")
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _load_index(self):
        index_path = "model_artifacts/item_index.faiss"
        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index not found. Run 'python -m src.utils' first!")
        self.index = faiss.read_index(index_path)

    def get_recommendations(self, user_id, k=10):
        """
        Full Inference Pipeline:
        User ID -> User Embedding -> Nearest Neighbor Search -> Item IDs
        """
        # 1. Prepare User Tensor
        if user_id >= self.cfg['model']['num_users']:
            # Simple cold-start handling (return generic items if user unknown)
            return [] 
            
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(self.device)
        
        # 2. Generate User Embedding
        with torch.no_grad():
            user_emb = self.model.user_embedding(user_tensor)
            user_vector = self.model.user_fc(user_emb)
            user_vector = torch.nn.functional.normalize(user_vector, p=2, dim=1)
            
        user_vector_np = user_vector.cpu().numpy()
        
        # 3. Search Index
        # D = Distances (Similarity Scores), I = Indices (Item IDs)
        D, I = self.index.search(user_vector_np, k)
        
        # 4. Format Response
        recommendations = []
        for score, item_id in zip(D[0], I[0]):
            recommendations.append({"item_id": int(item_id), "score": float(score)})
            
        return recommendations