"""
Remove unused YAML and SH files.

This script identifies and removes YAML config files and shell scripts
that are not referenced in the codebase.
"""

import os
import re
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).parent

# Config files that are definitely used (referenced in code)
USED_CONFIGS = {
    'configs/default.yaml',  # Default config in train.py
    'configs/db.yaml',  # Database config (might be used)
}

# Keep docker-compose.yml
KEEP_FILES = {
    'docker-compose.yml',
    'configs/default.yaml',
    'configs/db.yaml',
}

# Find all YAML files
def find_yaml_files():
    """Find all YAML/YML files."""
    yaml_files = []
    for root, dirs, filenames in os.walk(BASE_DIR):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in filenames:
            if filename.endswith(('.yaml', '.yml')):
                filepath = Path(root) / filename
                rel_path = filepath.relative_to(BASE_DIR)
                yaml_files.append(rel_path)
    
    return yaml_files

# Find all SH files
def find_sh_files():
    """Find all shell script files."""
    sh_files = []
    for root, dirs, filenames in os.walk(BASE_DIR):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in filenames:
            if filename.endswith('.sh'):
                filepath = Path(root) / filename
                rel_path = filepath.relative_to(BASE_DIR)
                sh_files.append(rel_path)
    
    return sh_files

# Check if file is referenced in code
def is_file_referenced(file_path: Path, search_in_code=True):
    """Check if a file is referenced in the codebase."""
    filename = file_path.name
    rel_path_str = str(file_path.relative_to(BASE_DIR))
    
    # Check if in keep list
    if rel_path_str in KEEP_FILES:
        return True
    
    # Search in Python files
    if search_in_code:
        try:
            # Use grep to search for references
            result = subprocess.run(
                ['grep', '-r', '--include=*.py', filename, str(BASE_DIR)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
            
            # Also search for path references
            result = subprocess.run(
                ['grep', '-r', '--include=*.py', rel_path_str.replace('/', '/'), str(BASE_DIR)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except:
            pass
    
    return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove unused YAML and SH files')
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete files (default: dry-run)')
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 FINDING UNUSED YAML AND SH FILES")
    print("="*80)
    
    # Find all files
    yaml_files = find_yaml_files()
    sh_files = find_sh_files()
    
    print(f"\n📄 Found {len(yaml_files)} YAML files")
    print(f"🔧 Found {len(sh_files)} SH files")
    
    # Check which are unused
    unused_yaml = []
    unused_sh = []
    
    print("\n📄 Checking YAML files...")
    for yaml_file in yaml_files:
        file_path = BASE_DIR / yaml_file
        if not is_file_referenced(file_path):
            unused_yaml.append(yaml_file)
            print(f"   ❌ Unused: {yaml_file}")
        else:
            print(f"   ✅ Used: {yaml_file}")
    
    print("\n🔧 Checking SH files...")
    for sh_file in sh_files:
        file_path = BASE_DIR / sh_file
        if not is_file_referenced(file_path):
            unused_sh.append(sh_file)
            print(f"   ❌ Unused: {sh_file}")
        else:
            print(f"   ✅ Used: {sh_file}")
    
    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(f"\nUnused YAML files: {len(unused_yaml)}")
    print(f"Unused SH files: {len(unused_sh)}")
    print(f"Total to delete: {len(unused_yaml) + len(unused_sh)}")
    
    if unused_yaml:
        print("\n📄 Unused YAML files:")
        for f in unused_yaml:
            print(f"   - {f}")
    
    if unused_sh:
        print("\n🔧 Unused SH files:")
        for f in unused_sh:
            print(f"   - {f}")
    
    # Delete if requested
    if args.execute:
        print("\n⚠️  WARNING: This will permanently delete files!")
        response = input("Type 'yes' to confirm: ")
        if response.lower() == 'yes':
            deleted = 0
            errors = 0
            
            for file_path in unused_yaml + unused_sh:
                full_path = BASE_DIR / file_path
                try:
                    if full_path.exists():
                        full_path.unlink()
                        deleted += 1
                        print(f"   ✅ Deleted: {file_path}")
                except Exception as e:
                    errors += 1
                    print(f"   ❌ Error deleting {file_path}: {e}")
            
            print(f"\n✅ Deletion complete!")
            print(f"   Deleted: {deleted} files")
            print(f"   Errors: {errors}")
        else:
            print("❌ Cancelled")
    else:
        print("\n💡 To actually delete, run:")
        print("   python remove_unused_yaml_sh.py --execute")

if __name__ == '__main__':
    main()

