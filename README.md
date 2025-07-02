# Cursor + Google Colab + GitHub Workflow

A complete workflow setup that lets you write code comfortably in Cursor while leveraging Google Colab Pro's GPU resources for computation.

## 🚀 Quick Start

### Step 1: Local Development Environment (Cursor)

1. **Clone or create project**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Add remote repository** (for new projects)
   ```bash
   git init
   git remote add origin https://github.com/your-username/your-repo.git
   ```

3. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

### Step 2: Google Colab Setup

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Set GPU runtime**: Runtime → Change runtime type → GPU

3. **First-time environment setup** (run in Colab):
   ```python
   # Clone your GitHub repository
   !git clone https://github.com/your-username/your-repo.git
   %cd your-repo
   
   # Install dependencies
   !pip install -r requirements.txt
   ```

4. **Subsequent usage** (after each code modification):
   ```python
   # Pull latest changes
   !git pull
   
   # Run your script
   !python main.py
   ```

## 📁 Project Structure

```
your-repo/
├── main.py              # Main machine learning script
├── colab_setup.py       # Colab environment setup script
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── .gitignore          # Git ignore file
```

## 🔧 Workflow

1. **Write code in Cursor** → Enjoy intelligent code completion and debugging
2. **Push to GitHub** → Version control and code synchronization
3. **Pull and run in Colab** → Utilize free GPU resources

```bash
# Complete workflow in Cursor
git add .
git commit -m "Update model architecture"
git push origin main
```

```python
# Complete workflow in Colab
!git pull
!python main.py
```

## 💡 Usage Tips

### Quick Sync Script

Create a quick push script in Cursor:
```bash
#!/bin/bash
git add .
git commit -m "$(date): Auto commit"
git push origin main
echo "✅ Code pushed to GitHub"
```

### Colab Optimization Tips

1. **Check GPU status**:
   ```python
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

2. **Monitor GPU memory**:
   ```python
   import torch
   print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
   print(f"GPU memory total: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
   ```

3. **Clear GPU memory**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## 🛠️ Common Issues

### Q: How to handle large files?
A: Use Git LFS or store large files in Google Drive:
```python
# Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')
```

### Q: How to save training results?
A: Save results to GitHub or Google Drive:
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Push back to GitHub (requires Colab GitHub authentication)
!git add model.pth
!git commit -m "Save trained model"
!git push
```

### Q: What if Colab session disconnects?
A: Re-run the environment setup code to restore.

## 📊 Example Project

The project includes a complete PyTorch example:
- Simple neural network
- GPU-accelerated training
- Training process visualization
- Automatic environment detection

## 🔗 Related Links

- [Google Colab](https://colab.research.google.com)
- [Cursor IDE](https://cursor.sh)
- [GitHub](https://github.com)

---

🎉 Now you can enjoy the perfect combination of local coding + cloud GPU! 