#!/usr/bin/env python3
"""
示例GPU加速的机器学习脚本
在Google Colab中使用GPU运行
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU不可用，使用CPU")
    return device

class SimpleNN(nn.Module):
    """简单的神经网络示例"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 784)  # 展平输入
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(device):
    """训练模型示例"""
    print("🚀 开始训练模型...")
    
    # 创建模拟数据
    batch_size = 64
    input_size = 784
    num_classes = 10
    num_epochs = 5
    
    # 创建模型并移动到GPU
    model = SimpleNN(input_size, 128, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练数据
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 100
        
        for batch in range(num_batches):
            # 生成随机数据
            data = torch.randn(batch_size, input_size).to(device)
            targets = torch.randint(0, num_classes, (batch_size,)).to(device)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return train_losses

def plot_results(losses):
    """绘制训练结果"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('Training Loss Over Time', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    """主函数"""
    print("=" * 50)
    print("🎯 Cursor + Colab + GitHub 工作流示例")
    print("=" * 50)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查GPU
    device = check_gpu()
    
    # 训练模型
    losses = train_model(device)
    
    # 绘制结果
    plot_results(losses)
    
    print("✅ 训练完成！")
    print("💡 提示: 修改代码后，在Cursor中推送到GitHub，然后在Colab中运行 !git pull 拉取最新代码")

if __name__ == "__main__":
    main() 