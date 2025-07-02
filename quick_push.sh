#!/bin/bash

# Quick push script - Simplify Git operations
# Usage: ./quick_push.sh "commit message"

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Starting to push code to GitHub...${NC}"

# Check if commit message is provided
if [ -z "$1" ]; then
    COMMIT_MSG="$(date '+%Y-%m-%d %H:%M:%S'): Auto commit"
    echo -e "${YELLOW}⚠️  No commit message provided, using default: $COMMIT_MSG${NC}"
else
    COMMIT_MSG="$1"
    echo -e "${GREEN}📝 Commit message: $COMMIT_MSG${NC}"
fi

# Add all files
echo -e "${YELLOW}📁 Adding files...${NC}"
git add .

# Check if any files were added
if git diff --staged --quiet; then
    echo -e "${YELLOW}⚠️  No file changes detected${NC}"
    exit 0
fi

# Commit
echo -e "${YELLOW}💾 Committing files...${NC}"
git commit -m "$COMMIT_MSG"

# Push
echo -e "${YELLOW}⬆️  Pushing to GitHub...${NC}"
if git push; then
    echo -e "${GREEN}✅ Code successfully pushed to GitHub!${NC}"
    echo -e "${GREEN}💡 Now you can run !git pull in Colab to get the latest code${NC}"
else
    echo -e "${RED}❌ Push failed, please check network connection and repository permissions${NC}"
    exit 1
fi 