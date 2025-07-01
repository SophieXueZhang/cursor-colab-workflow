#!/bin/bash

# 快速推送脚本 - 简化Git操作
# 使用方法: ./quick_push.sh "提交信息"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 开始推送代码到GitHub...${NC}"

# 检查是否提供了提交信息
if [ -z "$1" ]; then
    COMMIT_MSG="$(date '+%Y-%m-%d %H:%M:%S'): 自动提交"
    echo -e "${YELLOW}⚠️  未提供提交信息，使用默认信息: $COMMIT_MSG${NC}"
else
    COMMIT_MSG="$1"
    echo -e "${GREEN}📝 提交信息: $COMMIT_MSG${NC}"
fi

# 添加所有文件
echo -e "${YELLOW}📁 添加文件...${NC}"
git add .

# 检查是否有文件被添加
if git diff --staged --quiet; then
    echo -e "${YELLOW}⚠️  没有检测到文件变更${NC}"
    exit 0
fi

# 提交
echo -e "${YELLOW}💾 提交文件...${NC}"
git commit -m "$COMMIT_MSG"

# 推送
echo -e "${YELLOW}⬆️  推送到GitHub...${NC}"
if git push; then
    echo -e "${GREEN}✅ 代码成功推送到GitHub!${NC}"
    echo -e "${GREEN}💡 现在可以在Colab中运行 !git pull 来获取最新代码${NC}"
else
    echo -e "${RED}❌ 推送失败，请检查网络连接和仓库权限${NC}"
    exit 1
fi 