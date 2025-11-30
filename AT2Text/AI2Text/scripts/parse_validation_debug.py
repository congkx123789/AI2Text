#!/usr/bin/env python3
"""
Script để parse validation debug output từ training log.
Extract thông tin về predictions để debug WER=1.0.
"""

import re
import sys
from pathlib import Path


def parse_validation_debug(log_file: str):
    """
    Parse validation debug output từ log file.
    
    Args:
        log_file: Đường dẫn đến log file
    """
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"❌ Log file không tồn tại: {log_file}")
        return
    
    print("=" * 80)
    print("PARSE VALIDATION DEBUG OUTPUT")
    print("=" * 80)
    print()
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Tìm validation debug section
    debug_pattern = r"🔍 DEBUG: First validation batch predictions.*?VALIDATION SUMMARY"
    debug_match = re.search(debug_pattern, content, re.DOTALL)
    
    if not debug_match:
        print("⚠️  Không tìm thấy validation debug output trong log")
        print("   Có thể validation chưa chạy hoặc log chưa có đủ thông tin")
        print()
        print("   Đang tìm validation summary...")
        
        # Tìm validation summary
        summary_pattern = r"📊 VALIDATION SUMMARY.*?Empty predictions:.*?Blank-only predictions:.*?Valid outputs:"
        summary_match = re.search(summary_pattern, content, re.DOTALL)
        
        if summary_match:
            print("✅ Tìm thấy validation summary:")
            print()
            print(summary_match.group(0))
        else:
            print("❌ Không tìm thấy validation summary")
            print()
            print("   Kiểm tra log file có validation output không:")
            print(f"   tail -200 {log_file} | grep -i validation")
        
        return
    
    debug_section = debug_match.group(0)
    
    # Extract samples
    sample_pattern = r"Sample (\d+)/\d+:(.*?)(?=Sample \d+|=+ VALIDATION SUMMARY)"
    samples = re.findall(sample_pattern, debug_section, re.DOTALL)
    
    print(f"Tìm thấy {len(samples)} samples trong validation debug")
    print()
    
    # Parse từng sample
    for sample_num, sample_content in samples:
        print("=" * 80)
        print(f"Sample {sample_num}:")
        print("=" * 80)
        
        # Extract reference
        ref_match = re.search(r"📝 Reference:.*?Text: '([^']*)'.*?Length: (\d+) tokens", sample_content, re.DOTALL)
        if ref_match:
            ref_text = ref_match.group(1)
            ref_len = ref_match.group(2)
            print(f"Reference: '{ref_text}' ({ref_len} tokens)")
        
        # Extract prediction
        pred_match = re.search(r"🤖 Prediction:.*?Text: '([^']*)'.*?Length: (\d+) tokens", sample_content, re.DOTALL)
        if pred_match:
            pred_text = pred_match.group(1)
            pred_len = pred_match.group(2)
            print(f"Prediction: '{pred_text}' ({pred_len} tokens)")
        
        # Extract analysis
        analysis_match = re.search(r"📊 Analysis:.*?Unique pred tokens: (.*?)\n.*?Number of unique tokens: (\d+)", sample_content, re.DOTALL)
        if analysis_match:
            unique_tokens = analysis_match.group(1)
            num_unique = analysis_match.group(2)
            print(f"Unique tokens: {unique_tokens} ({num_unique} unique)")
        
        # Check warnings
        if "🚨 PREDICTION IS EMPTY" in sample_content:
            print("🚨 PREDICTION IS EMPTY!")
        elif "🚨 ALL PREDICTIONS ARE BLANK TOKEN" in sample_content:
            print("🚨 ALL PREDICTIONS ARE BLANK TOKEN!")
        elif "⚠️  VERY FEW UNIQUE TOKENS" in sample_content:
            print("⚠️  VERY FEW UNIQUE TOKENS")
        else:
            print("✅ Prediction có nội dung")
        
        print()
    
    # Extract summary
    summary_pattern = r"📊 VALIDATION SUMMARY.*?Empty predictions: (\d+) \((\d+\.\d+)%\).*?Blank-only predictions: (\d+) \((\d+\.\d+)%\).*?Valid outputs: (\d+) \((\d+\.\d+)%\)"
    summary_match = re.search(summary_pattern, content, re.DOTALL)
    
    if summary_match:
        print("=" * 80)
        print("VALIDATION SUMMARY:")
        print("=" * 80)
        print(f"Empty predictions: {summary_match.group(1)} ({summary_match.group(2)}%)")
        print(f"Blank-only predictions: {summary_match.group(3)} ({summary_match.group(4)}%)")
        print(f"Valid outputs: {summary_match.group(5)} ({summary_match.group(6)}%)")
        print()
        
        empty_pct = float(summary_match.group(2))
        blank_pct = float(summary_match.group(4))
        
        if empty_pct > 50 or blank_pct > 50:
            print("🚨 CẢNH BÁO: Hơn 50% predictions là empty/blank!")
            print("   Model có thể đã collapse hoặc chưa học được gì")
        else:
            print("✅ Model có output đa dạng")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 parse_validation_debug.py <log_file>")
        print("Example: python3 parse_validation_debug.py logs/test_loss_training.log")
        sys.exit(1)
    
    parse_validation_debug(sys.argv[1])

