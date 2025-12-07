#!/bin/bash

# Script to remove large files from git history
# This will remove the large CSV and JSON files that exceed GitHub's limits

echo "Removing large files from git history..."
echo "This may take a while..."

# List of large files to remove from git history
LARGE_FILES=(
    "ai-llm-ss/data/processed/full_merged_dataset/train/timestamps.json"
    "ai-llm-ss/data/processed/merged_dataset/train/timestamps.json"
    "ai-llm-ss/data/processed/full_merged_dataset/train/manifest.csv"
    "ai-llm-ss/data/processed/merged_dataset/train/manifest.csv"
    "ai-llm-ss/data/processed/merged_dataset/train/manifest_sliced.csv"
    "ai-llm-ss/data/processed/merged_dataset/train/manifest_sorted.csv"
    "ai-llm-ss/data/processed/full_merged_dataset/test/timestamps.json"
    "ai-llm-ss/data/processed/full_merged_dataset/val/timestamps.json"
)

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "git-filter-repo is not installed. Installing..."
    pip install git-filter-repo
fi

# Remove each file from git history
for file in "${LARGE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file from git history..."
        git filter-repo --path "$file" --invert-paths --force
    else
        echo "File $file not found, removing from history anyway..."
        git filter-repo --path "$file" --invert-paths --force
    fi
done

echo ""
echo "Done! Large files have been removed from git history."
echo ""
echo "Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Force push to remote: git push origin main --force"
echo "   (WARNING: This will rewrite history on remote. Make sure you coordinate with your team!)"
echo ""
echo "Note: The files still exist locally but are now ignored by .gitignore"

