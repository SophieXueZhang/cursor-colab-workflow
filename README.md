# Cursor + Google Colab + GitHub Workflow

A streamlined workflow setup that lets you write code comfortably in Cursor while leveraging Google Colab's GPU resources for computation.

## 🚀 Quick Start

### Step 1: Local Development Environment (Cursor)

1. **Clone or create project**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

### Step 2: Google Colab Setup

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Set GPU runtime**: Runtime → Change runtime type → GPU

3. **Open the provided notebook** `colab_example.ipynb` or run the setup cell:
   ```python
   # 🔧 Setup - Run this first!
   REPO_URL = "https://github.com/SophieXueZhang/cursor-colab-workflow-en.git"
   REPO_NAME = REPO_URL.split("/")[-1].replace(".git", "")
   
   import os, shutil
   if os.path.exists(REPO_NAME): shutil.rmtree(REPO_NAME)
   !git clone $REPO_URL
   %cd $REPO_NAME
   %pip install -r requirements.txt
   print("✅ Setup complete!")
   ```

4. **Update and run** (after each code modification):
   ```python
   # Update code
   !git pull
   
   # Run your script
   !python main.py
   ```

## 📁 Project Structure

```
cursor-colab-workflow-en/
├── README.md           # Project documentation
├── main.py             # Simple PyTorch GPU training example  
├── requirements.txt    # Minimal dependencies: torch, matplotlib, numpy
└── colab_example.ipynb # Ready-to-use Colab notebook
```

## 🔧 Workflow

1. **Write code in Cursor** → Enjoy intelligent code completion and debugging
2. **Push to GitHub** → Version control and code synchronization  
3. **Pull and run in Colab** → Utilize free GPU resources

```bash
# Complete workflow in Cursor
git add .
git commit -m "Update model"
git push origin main
```

```python
# Complete workflow in Colab
!git pull
!python main.py
```

## 💡 Usage Tips

### Git Commands in Cursor

Use standard git commands for version control:
```bash
git add .
git commit -m "Your commit message"  
git push origin main
```

### Colab Optimization Tips

1. **Check GPU status**:
   ```python
   import torch
   print(f"GPU: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"Name: {torch.cuda.get_device_name(0)}")
   ```

2. **Quick commands**:
   ```python
   !git pull      # Update code
   !python main.py # Run training
   ```

## 🛠️ Common Issues

### Q: How to change repository?
A: Simply modify the `REPO_URL` variable in the setup cell.

### Q: What if Colab session disconnects?
A: Re-run the setup cell to restore your environment.

### Q: How to save training results?
A: Results are automatically plotted. For permanent storage, save to Google Drive or push back to GitHub.

## 📊 Example Project

The project includes a complete PyTorch example:
- Simple neural network (66 lines total)
- GPU-accelerated training
- Training loss visualization
- Automatic device detection

## 🔗 Related Links

- [Google Colab](https://colab.research.google.com)
- [Cursor IDE](https://cursor.sh)
- [GitHub](https://github.com)

---

🎉 Now you can enjoy the perfect combination of local coding + cloud GPU with minimal setup! 