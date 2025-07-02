"""
Simple Colab setup for cursor-colab-workflow-en
"""

def setup():
    """Quick setup"""
    import os, shutil
    repo_url = "https://github.com/SophieXueZhang/cursor-colab-workflow-en.git"
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    
    print(f"📂 Repository: {repo_name}")
    print(f"🔗 URL: {repo_url}")
    
    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)
        print(f"🗑️ Removed existing {repo_name}")
    
    print("📥 Cloning repository...")
    result = os.system(f"git clone {repo_url}")
    if result == 0:
        print("✅ Clone successful!")
    else:
        print("❌ Clone failed!")
        return
    
    print("📁 Changing directory...")
    try:
        os.chdir(repo_name)
        print(f"Current directory: {os.getcwd()}")
    except Exception as e:
        print(f"❌ Failed to change directory: {e}")
        return
    
    print("📦 Installing dependencies...")
    result = os.system("pip install -r requirements.txt")
    if result == 0:
        print("✅ Dependencies installed!")
    else:
        print("❌ Installation failed!")
        return
        
    print("✅ Setup complete!")

def update():
    """Pull latest changes"""
    import os
    print("🔄 Pulling latest changes...")
    result = os.system("git pull")
    if result == 0:
        print("✅ Updated!")
    else:
        print("❌ Update failed!")

def setup_colab():
    """Setup specifically for Colab with magic commands"""
    print("Use this in a Colab cell instead:")
    print("""
# 🔧 Setup - Run this in Colab!
REPO_URL = "https://github.com/SophieXueZhang/cursor-colab-workflow-en.git"
REPO_NAME = REPO_URL.split("/")[-1].replace(".git", "")

import os, shutil
if os.path.exists(REPO_NAME):
    shutil.rmtree(REPO_NAME)

!git clone $REPO_URL
%cd $REPO_NAME
%pip install -r requirements.txt
print("✅ Setup complete!")
""")

# Usage:
# For Colab: use setup_colab() to get the correct commands
# For regular Python: setup() then update()
# setup_colab()  # Shows Colab commands
# setup()        # For regular Python
# update()       # Pull updates 