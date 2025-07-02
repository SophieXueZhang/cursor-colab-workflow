#!/bin/bash
# Simple git push script

MSG=${1:-"Auto update $(date '+%H:%M')"}

git add .
git commit -m "$MSG"
git push

echo "✅ Pushed: $MSG" 