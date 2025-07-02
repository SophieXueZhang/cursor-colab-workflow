#!/usr/bin/env python3
"""
Example GPU-accelerated machine learning script
Run with GPU in Google Colab
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
    return device

class SimpleNN(nn.Module):
    """Simple neural network example"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(device):
    """Train model example"""
    print("🚀 Starting model training...")
    
    # GPU memory monitoring
    if torch.cuda.is_available():
        print(f"💾 GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Create mock data
    batch_size = 64
    input_size = 784
    num_classes = 10
    num_epochs = 5
    
    # Create model and move to GPU
    model = SimpleNN(input_size, 128, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # GPU memory monitoring
    if torch.cuda.is_available():
        print(f"💾 GPU memory after model loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    # Mock training data
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 100
        
        for batch in range(num_batches):
            # Generate random data
            data = torch.randn(batch_size, input_size).to(device)
            targets = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Show GPU memory usage after each epoch
        if torch.cuda.is_available():
            print(f"  💾 GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    return train_losses

def plot_results(losses):
    """Plot training results"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Time', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """Main function"""
    print("=" * 50)
    print("🎯 Cursor + Colab + GitHub Workflow Example")
    print("=" * 50)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔥 This is a test update - demonstrating Cursor to Colab workflow")
    print("✨ User test modification - verifying workflow runs properly")
    
    # Check GPU
    device = check_gpu()
    
    # Train model
    losses = train_model(device)
    
    # Plot results
    plot_results(losses)
    
    print("✅ Training completed!!!")
    print("💡 Tip: After modifying code, push to GitHub in Cursor, then run !git pull in Colab to get latest code")

if __name__ == "__main__":
    main() 