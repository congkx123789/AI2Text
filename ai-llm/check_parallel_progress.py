"""
Check progress của parallel GPU test
"""
import json
import glob
from pathlib import Path

def check_progress():
    """Check progress của tất cả parallel processes"""
    files = glob.glob('test_results_parallel/whisper_test_bf16_part*.json')
    
    if not files:
        print("❌ No result files found")
        return
    
    print("=" * 80)
    print("📊 PARALLEL GPU TEST PROGRESS")
    print("=" * 80)
    
    total_samples = 0
    total_wer = 0
    total_cer = 0
    perfect_matches = 0
    
    for file in sorted(files):
        try:
            data = json.load(open(file))
            results = data.get('results', [])
            samples = len(results)
            total_samples += samples
            
            if results:
                avg_wer = sum(r['wer'] for r in results) / len(results)
                avg_cer = sum(r['cer'] for r in results) / len(results)
                perfect = sum(1 for r in results if r['match'])
                
                total_wer += avg_wer * samples
                total_cer += avg_cer * samples
                perfect_matches += perfect
                
                print(f"\n📁 {Path(file).name}:")
                print(f"   Samples: {samples}")
                print(f"   Avg WER: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
                print(f"   Avg CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
                print(f"   Perfect: {perfect}/{samples} ({perfect/samples*100:.2f}%)")
        except Exception as e:
            print(f"⚠️  Error reading {file}: {e}")
    
    print("\n" + "=" * 80)
    print("📈 OVERALL PROGRESS")
    print("=" * 80)
    print(f"Total samples: {total_samples}/32818 ({total_samples/32818*100:.2f}%)")
    
    if total_samples > 0:
        overall_wer = total_wer / total_samples
        overall_cer = total_cer / total_samples
        perfect_rate = perfect_matches / total_samples
        
        print(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
        print(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
        print(f"Perfect matches: {perfect_matches}/{total_samples} ({perfect_rate*100:.2f}%)")
    
    print("=" * 80)

if __name__ == "__main__":
    check_progress()

