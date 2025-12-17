import torch
import torch.nn as nn
import torch.optim as optim
from tlsan_model import TLSAN

def train_tlsan(dataloader, num_items, epochs=5):
    device = torch.device("cpu")
    model = TLSAN(num_items=num_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequences, pos_items, neg_items in dataloader:
            sequences = sequences.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            pos_scores = model(sequences, pos_items)
            neg_scores = model(sequences, neg_items)

            labels = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores)
            ])
            preds = torch.cat([pos_scores, neg_scores])

            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")