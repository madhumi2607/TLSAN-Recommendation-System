import torch
import torch.nn as nn

class TLSAN(nn.Module):
    def __init__(self, num_items, embedding_dim=64, seq_len=5):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=1, batch_first=True
        )
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, sequences, candidates):
        seq_emb = self.item_embedding(sequences)
        positions = torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        x = seq_emb + pos_emb
        attn_output, _ = self.self_attention(x, x, x)
        seq_repr = attn_output.mean(dim=1)

        cand_emb = self.item_embedding(candidates)
        scores = (self.fc(seq_repr) * cand_emb).sum(dim=1)
        return scores