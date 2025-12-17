import torch
import torch.nn as nn
import torch.nn.functional as F

class PACA(nn.Module):
    def __init__(self, num_items, embed_dim=64, sequence_length=5):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.position_embedding = nn.Embedding(sequence_length, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.attention = nn.Linear(embed_dim * 2, 1)

    def forward(self, seq, target):
        seq_emb = self.item_embedding(seq)
        pos_ids = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos_ids)

        x = seq_emb + pos_emb
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        concat = torch.cat([x, pos_emb.expand_as(x)], dim=-1)
        attn_weights = F.softmax(self.attention(concat).squeeze(-1), dim=-1)
        context = torch.sum(attn_weights.unsqueeze(-1) * x, dim=1)

        target_emb = self.item_embedding(target)
        scores = torch.sum(context * target_emb, dim=-1)
        return scores