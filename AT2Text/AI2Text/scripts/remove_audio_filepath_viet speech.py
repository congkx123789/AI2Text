#!/usr/bin/env python3
"""
Remove audio_filepath field from VietSpeech JSON timestamp files.
"""

import json
import os
from pathlib import Path


def remove_audio_filepath_from_json(json_path):
    """
    Remove 'audio_filepath' field from all entries in a JSON file.
    
    Args:
        json_path: Path to the JSON file
    """
    print(f"Processing: {json_path}")
    
    # Check if file exists
    if not os.path.exists(json_path):
        print(f"  File not found: {json_path}")
        return False
    
    # Get file size for progress indication
    file_size = os.path.getsize(json_path) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")
    
    try:
        # Read the JSON file
        print("  Reading JSON file...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Count entries before processing
        total_entries = len(data)
        print(f"  Total entries: {total_entries}")
        
        # Remove audio_filepath from each entry
        removed_count = 0
        for key, entry in data.items():
            if isinstance(entry, dict) and 'audio_filepath' in entry:
                del entry['audio_filepath']
                removed_count += 1
        
        print(f"  Removed 'audio_filepath' from {removed_count} entries")
        
        # Write back to file
        print("  Writing updated JSON file...")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Successfully updated: {json_path}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error processing file: {e}")
        return False


def main():
    """Main function to process all VietSpeech JSON files."""
    base_dir = Path("/home/alida/Documents/Cursor/AI2Text/AT2Text/AI2Text/data/processed/VietSpeech")
    
    # JSON files to process
    json_files = [
        base_dir / "train" / "timestamps.json",
        base_dir / "val" / "timestamps.json",
        base_dir / "test" / "timestamps.json",
    ]
    
    print("=" * 60)
    print("Removing 'audio_filepath' from VietSpeech JSON files")
    print("=" * 60)
    print()
    
    success_count = 0
    for json_file in json_files:
        if remove_audio_filepath_from_json(json_file):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"Processing complete: {success_count}/{len(json_files)} files updated")
    print("=" * 60)


if __name__ == "__main__":
    main()

