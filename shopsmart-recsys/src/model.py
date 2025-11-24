import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dim=32):
        super(TwoTowerModel, self).__init__()
        
        # --- User Tower ---
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) 
        )
        
        # --- Item Tower ---
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, user_ids, item_ids):
        # Pass inputs through respective towers
        u_emb = self.user_embedding(user_ids)
        u_vec = self.user_fc(u_emb)
        
        i_emb = self.item_embedding(item_ids)
        i_vec = self.item_fc(i_emb)
        
        # Normalize vectors for Cosine Similarity
        u_vec = F.normalize(u_vec, p=2, dim=1)
        i_vec = F.normalize(i_vec, p=2, dim=1)
        
        # Dot Product
        score = (u_vec * i_vec).sum(dim=1)
        
        return score

    def get_user_embedding(self, user_ids):
        """Used during inference"""
        u_emb = self.user_embedding(user_ids)
        return F.normalize(self.user_fc(u_emb), p=2, dim=1)