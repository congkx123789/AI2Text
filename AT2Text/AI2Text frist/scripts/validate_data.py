"""
Standalone script to validate and check database readiness for training.
"""

import argparse
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from database.db_utils import ASRDatabase
import pandas as pd


def print_data_summary(db: ASRDatabase, split_version: str = "v1"):
    """Print comprehensive data summary."""
    summary = db.get_data_summary(split_version)
    
    print("\n" + "="*70)
    print("DATABASE DATA SUMMARY")
    print("="*70)
    
    overall = summary['overall']
    print("\n[Overall Statistics]")
    print(f"  Total files:           {overall['total_files']:,}")
    print(f"  Total speakers:         {overall['total_speakers']:,}")
    print(f"  Total duration:         {overall['total_duration_hours']:.2f} hours")
    print(f"  Average duration:        {overall['avg_duration']:.2f} seconds")
    print(f"  Avg words/transcript:   {overall['avg_words_per_transcript']:.1f}")
    
    print("\n[Split Statistics]")
    for split_type, stats in summary['splits'].items():
        print(f"\n  {split_type.upper()}:")
        print(f"    Files:               {stats.get('num_files', 0):,}")
        print(f"    Duration:             {stats.get('total_duration_seconds', 0) / 3600:.2f} hours")
        print(f"    Avg duration:         {stats.get('avg_duration_seconds', 0):.2f} seconds")
        print(f"    Speakers:             {stats.get('num_speakers', 0)}")
    
    print("\n[Validation Results]")
    validation = summary['validation']
    
    if validation['is_ready']:
        print("  [OK] Database is READY for training!")
    else:
        print("  [ERROR] Database is NOT ready:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    
    if validation['warnings']:
        print("\n  [WARNING] Warnings:")
        for warning in validation['warnings']:
            print(f"    - {warning}")
    
    if validation['recommendations']:
        print("\n  [TIP] Recommendations:")
        for rec in validation['recommendations']:
            print(f"    - {rec}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Validate database for training')
    parser.add_argument('--config', type=str, default='configs/db.yaml',
                       help='Path to database config')
    parser.add_argument('--export', type=str, default=None,
                       help='Export summary to CSV file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize database
    db = ASRDatabase(config['database']['path'])
    
    # Print summary
    print_data_summary(db, config['split_version'])
    
    # Export if requested
    if args.export:
        stats = db.get_dataset_statistics(config['split_version'])
        stats.to_csv(args.export, index=False, encoding='utf-8')
        print(f"\n[OK] Statistics exported to: {args.export}")


if __name__ == '__main__':
    main()

