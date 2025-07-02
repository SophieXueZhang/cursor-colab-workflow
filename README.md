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

3. **Run the setup cell** (in Colab):
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
your-repo/
├── main.py              # Simple PyTorch GPU training example
├── colab_example.ipynb  # Ready-to-use Colab notebook
├── colab_setup.py       # Python setup functions
├── quick_push.sh        # One-line git push script
├── requirements.txt     # Minimal dependencies: torch, matplotlib, numpy
└── README.md           # Project documentation
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

# Or use the quick script
./quick_push.sh "Update model"
```

```python
# Complete workflow in Colab
!git pull
!python main.py
```

## 💡 Usage Tips

### Quick Push Script

Use the simplified push script in Cursor:
```bash
./quick_push.sh "your message"
```

### Colab Optimization Tips

1. **Check GPU status**:
   ```python
   import torch
   print(f"GPU: {torch.cuda.is_available()}")
   if torch.cuda.is_available():
       print(f"Name: {torch.cuda.get_device_name(0)}")
   ```

2. **Quick setup function**:
   ```python
   from colab_setup import setup, update
   setup()  # First time
   update() # After changes
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
- Simple neural network (52 lines vs 134 lines)
- GPU-accelerated training
- Training loss visualization
- Automatic device detection

## 🔗 Related Links

- [Google Colab](https://colab.research.google.com)
- [Cursor IDE](https://cursor.sh)
- [GitHub](https://github.com)

---

🎉 Now you can enjoy the perfect combination of local coding + cloud GPU with minimal setup! 