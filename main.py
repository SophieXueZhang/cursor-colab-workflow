#!/usr/bin/env python3
"""
Simple GPU training example for Cursor + Colab workflow
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    return device

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.layers(x.view(-1, 784))

def train_model(device):
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(5):
        epoch_loss = 0
        for _ in range(100):
            data = torch.randn(64, 784).to(device)
            targets = torch.randint(0, 10, (64,)).to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(data), targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 100
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: {avg_loss:.3f}")
    
    return losses

def main():
    print("🎯 Cursor + Colab Workflow Demo")
    device = check_gpu()
    losses = train_model(device)
    
    plt.plot(losses)
    plt.title('Training Loss')
    plt.show()
    
    print("✅ Done!")

if __name__ == "__main__":
    main() 