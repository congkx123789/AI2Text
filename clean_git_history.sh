#!/bin/bash

# Script to clean large files from git history
# This will remove large CSV and JSON files that exceed GitHub's limits

set -e

echo "=========================================="
echo "Cleaning large files from git history"
echo "=========================================="
echo ""

# Files to remove from git history
FILES_TO_REMOVE=(
    "ai-llm-ss/data/processed/full_merged_dataset/train/timestamps.json"
    "ai-llm-ss/data/processed/merged_dataset/train/timestamps.json"
    "ai-llm-ss/data/processed/full_merged_dataset/train/manifest.csv"
    "ai-llm-ss/data/processed/merged_dataset/train/manifest.csv"
    "ai-llm-ss/data/processed/merged_dataset/train/manifest_sliced.csv"
    "ai-llm-ss/data/processed/merged_dataset/train/manifest_sorted.csv"
    "ai-llm-ss/data/processed/full_merged_dataset/test/timestamps.json"
    "ai-llm-ss/data/processed/full_merged_dataset/val/timestamps.json"
)

# Build the git rm command
RM_CMD="git rm --cached --ignore-unmatch"
for file in "${FILES_TO_REMOVE[@]}"; do
    RM_CMD="$RM_CMD \"$file\""
done

echo "Removing files from git history..."
echo "This may take several minutes..."
echo ""

# Use git filter-branch to remove files from all commits
git filter-branch --force --index-filter "$RM_CMD" --prune-empty --tag-name-filter cat -- --all

echo ""
echo "Cleaning up..."
# Remove backup refs
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d 2>/dev/null || true

# Expire reflog
git reflog expire --expire=now --all

# Garbage collection
git gc --prune=now --aggressive

echo ""
echo "=========================================="
echo "Done! Large files have been removed from git history."
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Add the updated .gitignore: git add .gitignore"
echo "3. Commit: git commit -m 'Update .gitignore to exclude large data files'"
echo "4. Force push to remote:"
echo "   git push origin main --force"
echo ""
echo "⚠️  WARNING: Force push will rewrite history on remote!"
echo "   Make sure you coordinate with your team if working collaboratively."
echo ""
echo "The files still exist locally but are now ignored by .gitignore"

