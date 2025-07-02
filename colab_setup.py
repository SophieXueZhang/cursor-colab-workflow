"""
Google Colab environment setup script
Copy and run this code in Colab to set up the environment
"""

def setup_colab_environment(github_repo_url, repo_name):
    """
    Set up Colab environment
    
    Args:
        github_repo_url (str): GitHub repository URL, e.g.: "https://github.com/username/repo.git"
        repo_name (str): Repository name, e.g.: "repo"
    """
    
    print("🔧 Setting up Colab environment...")
    
    # 1. Clone repository
    print("📥 Cloning GitHub repository...")
    import os
    if os.path.exists(repo_name):
        print(f"⚠️  Directory {repo_name} already exists, removing...")
        import shutil
        shutil.rmtree(repo_name)
    
    os.system(f"git clone {github_repo_url}")
    os.chdir(repo_name)
    print(f"✅ Successfully cloned repository to {repo_name}")
    
    # 2. Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  GPU not available")
    
    # 3. Install dependencies
    print("📦 Installing dependencies...")
    if os.path.exists("requirements.txt"):
        os.system("pip install -r requirements.txt")
        print("✅ Dependencies installed successfully")
    else:
        print("⚠️  requirements.txt file not found")
    
    print("🎉 Environment setup complete!")
    print("\n📝 Usage instructions:")
    print("1. After modifying code in Cursor, run: git add . && git commit -m 'update' && git push")
    print("2. In Colab, run: !git pull to get the latest code")
    print("3. Run: !python main.py to execute your script")

def pull_latest_changes():
    """Pull the latest code changes"""
    print("🔄 Pulling latest code...")
    import os
    result = os.system("git pull")
    if result == 0:
        print("✅ Code updated successfully!")
    else:
        print("❌ Code update failed, please check network connection or repository status")

# Example usage (run in Colab):
"""
# 1. First-time setup (replace with your repository information)
setup_colab_environment(
    github_repo_url="https://github.com/your-username/your-repo.git",
    repo_name="your-repo"
)

# 2. Subsequent code updates
pull_latest_changes()

# 3. Run your script
!python main.py
""" 