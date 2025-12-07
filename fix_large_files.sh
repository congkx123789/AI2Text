#!/bin/bash

# Quick fix: Remove large files from git tracking (but keep them locally)
# This is simpler than rewriting history

echo "Removing large files from git tracking..."
echo ""

# Remove large files from git index (but keep them locally)
git rm --cached ai-llm-ss/data/processed/full_merged_dataset/train/timestamps.json 2>/dev/null
git rm --cached ai-llm-ss/data/processed/merged_dataset/train/timestamps.json 2>/dev/null
git rm --cached ai-llm-ss/data/processed/full_merged_dataset/train/manifest.csv 2>/dev/null
git rm --cached ai-llm-ss/data/processed/merged_dataset/train/manifest.csv 2>/dev/null
git rm --cached ai-llm-ss/data/processed/merged_dataset/train/manifest_sliced.csv 2>/dev/null
git rm --cached ai-llm-ss/data/processed/merged_dataset/train/manifest_sorted.csv 2>/dev/null
git rm --cached ai-llm-ss/data/processed/full_merged_dataset/test/timestamps.json 2>/dev/null
git rm --cached ai-llm-ss/data/processed/full_merged_dataset/val/timestamps.json 2>/dev/null

echo ""
echo "Files removed from git tracking."
echo ""
echo "Next steps:"
echo "1. Commit the removal: git commit -m 'Remove large files from tracking'"
echo "2. Push to remote: git push origin main"
echo ""
echo "Note: If these files are already in remote history, you may need to:"
echo "  - Use git filter-branch or BFG Repo-Cleaner to remove from history"
echo "  - Or create a new branch without these files"

