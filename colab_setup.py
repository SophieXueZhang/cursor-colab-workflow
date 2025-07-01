"""
Google Colab 环境设置脚本
在Colab中复制并运行此代码来设置环境
"""

def setup_colab_environment(github_repo_url, repo_name):
    """
    设置Colab环境
    
    Args:
        github_repo_url (str): GitHub仓库URL，例如: "https://github.com/username/repo.git"
        repo_name (str): 仓库名称，例如: "repo"
    """
    
    print("🔧 正在设置Colab环境...")
    
    # 1. 克隆仓库
    print("📥 克隆GitHub仓库...")
    import os
    if os.path.exists(repo_name):
        print(f"⚠️  目录 {repo_name} 已存在，正在删除...")
        import shutil
        shutil.rmtree(repo_name)
    
    os.system(f"git clone {github_repo_url}")
    os.chdir(repo_name)
    print(f"✅ 成功克隆仓库到 {repo_name}")
    
    # 2. 检查GPU
    import torch
    if torch.cuda.is_available():
        print(f"🚀 GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  GPU不可用")
    
    # 3. 安装依赖
    print("📦 安装依赖...")
    if os.path.exists("requirements.txt"):
        os.system("pip install -r requirements.txt")
        print("✅ 依赖安装完成")
    else:
        print("⚠️  未找到requirements.txt文件")
    
    print("🎉 环境设置完成！")
    print("\n📝 使用说明:")
    print("1. 在Cursor中修改代码后，运行: git add . && git commit -m 'update' && git push")
    print("2. 在Colab中运行: !git pull 来获取最新代码")
    print("3. 运行: !python main.py 来执行你的脚本")

def pull_latest_changes():
    """拉取最新的代码变更"""
    print("🔄 正在拉取最新代码...")
    import os
    result = os.system("git pull")
    if result == 0:
        print("✅ 代码更新成功!")
    else:
        print("❌ 代码更新失败，请检查网络连接或仓库状态")

# 示例使用方法（在Colab中运行）:
"""
# 1. 首次设置（替换为你的仓库信息）
setup_colab_environment(
    github_repo_url="https://github.com/your-username/your-repo.git",
    repo_name="your-repo"
)

# 2. 后续更新代码
pull_latest_changes()

# 3. 运行你的脚本
!python main.py
""" 