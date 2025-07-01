# Cursor + Google Colab + GitHub 工作流

这是一个完整的工作流设置，让你可以在Cursor中舒适地编写代码，同时利用Google Colab Pro的GPU资源进行计算。

## 🚀 快速开始

### 第一步：本地开发环境准备（Cursor）

1. **克隆或创建项目**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **添加远程仓库**（如果是新项目）
   ```bash
   git init
   git remote add origin https://github.com/your-username/your-repo.git
   ```

3. **推送代码到GitHub**
   ```bash
   git add .
   git commit -m "初始提交"
   git push -u origin main
   ```

### 第二步：Google Colab设置

1. **打开Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **设置GPU运行时**: Runtime → Change runtime type → GPU

3. **首次设置环境**（在Colab中运行）:
   ```python
   # 克隆你的GitHub仓库
   !git clone https://github.com/your-username/your-repo.git
   %cd your-repo
   
   # 安装依赖
   !pip install -r requirements.txt
   ```

4. **后续使用**（每次修改代码后）:
   ```python
   # 拉取最新修改
   !git pull
   
   # 运行你的脚本
   !python main.py
   ```

## 📁 项目结构

```
your-repo/
├── main.py              # 主要的机器学习脚本
├── colab_setup.py       # Colab环境设置脚本
├── requirements.txt     # Python依赖
├── README.md           # 项目说明
└── .gitignore          # Git忽略文件
```

## 🔧 工作流程

1. **在Cursor中编写代码** → 享受智能代码补全和调试
2. **推送到GitHub** → 版本控制和代码同步
3. **在Colab中拉取并运行** → 利用免费GPU资源

```bash
# 在Cursor中的完整工作流
git add .
git commit -m "更新模型架构"
git push origin main
```

```python
# 在Colab中的完整工作流
!git pull
!python main.py
```

## 💡 使用技巧

### 快速同步脚本

在Cursor中创建快速推送脚本：
```bash
#!/bin/bash
git add .
git commit -m "$(date): 自动提交"
git push origin main
echo "✅ 代码已推送到GitHub"
```

### Colab优化技巧

1. **检查GPU状态**:
   ```python
   import torch
   print(f"GPU可用: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU名称: {torch.cuda.get_device_name(0)}")
   ```

2. **监控GPU内存**:
   ```python
   import torch
   print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
   print(f"GPU内存总量: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
   ```

3. **清理GPU内存**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## 🛠️ 常见问题

### Q: 如何处理大文件？
A: 使用Git LFS或将大文件存储在Google Drive中：
```python
# 在Colab中挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

### Q: 如何保存训练结果？
A: 将结果保存到GitHub或Google Drive：
```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 推送回GitHub（需要配置Colab的GitHub认证）
!git add model.pth
!git commit -m "保存训练模型"
!git push
```

### Q: Colab会话断开怎么办？
A: 重新运行环境设置代码即可恢复。

## 📊 示例项目

项目包含一个完整的PyTorch示例：
- 简单的神经网络
- GPU加速训练
- 训练过程可视化
- 自动环境检测

## 🔗 相关链接

- [Google Colab](https://colab.research.google.com)
- [Cursor IDE](https://cursor.sh)
- [GitHub](https://github.com)

---

🎉 现在你可以享受本地编码 + 云端GPU的完美结合！ 