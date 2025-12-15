#!/usr/bin/env python3
"""
Script để kiểm tra sample rate của file audio WAV.
Cực kỳ quan trọng để đảm bảo audio được resample về 16kHz trước khi đưa vào model.
"""

import sys
from pathlib import Path
import torchaudio
import librosa
import numpy as np

def check_sample_rate(audio_path: str, verbose: bool = True) -> dict:
    """
    Kiểm tra sample rate của file audio.
    
    Args:
        audio_path: Đường dẫn đến file audio
        verbose: In thông tin chi tiết
        
    Returns:
        Dictionary chứa thông tin sample rate
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        print(f"❌ File không tồn tại: {audio_path}")
        return None
    
    result = {
        'file': str(audio_path),
        'exists': True,
        'torchaudio_sr': None,
        'librosa_sr': None,
        'is_16khz': False,
        'needs_resample': False,
        'duration': None,
        'channels': None,
        'format': None
    }
    
    # Kiểm tra với torchaudio (nhanh hơn)
    try:
        info = torchaudio.info(audio_path)
        result['torchaudio_sr'] = info.sample_rate
        result['channels'] = info.num_channels
        result['duration'] = info.num_frames / info.sample_rate
        result['format'] = 'torchaudio'
        
        if verbose:
            print(f"📁 File: {audio_path.name}")
            print(f"   Sample Rate: {info.sample_rate} Hz")
            print(f"   Channels: {info.num_channels}")
            print(f"   Duration: {result['duration']:.2f} seconds")
            print(f"   Frames: {info.num_frames:,}")
        
        # Kiểm tra xem có phải 16kHz không
        if info.sample_rate == 16000:
            result['is_16khz'] = True
            if verbose:
                print("   ✅ Sample rate đúng (16kHz)")
        else:
            result['needs_resample'] = True
            if verbose:
                print(f"   ⚠️  Sample rate KHÔNG đúng! Cần resample từ {info.sample_rate}Hz → 16kHz")
                print(f"   🚨 Nếu không resample, model sẽ nghe như 'người ngoài hành tinh'!")
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Không thể đọc với torchaudio: {e}")
            print("   → Thử với librosa...")
        
        # Fallback to librosa
        try:
            info = librosa.get_duration(filename=str(audio_path))
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            
            result['librosa_sr'] = sr
            result['duration'] = info
            result['format'] = 'librosa'
            
            if isinstance(audio, np.ndarray):
                if audio.ndim > 1:
                    result['channels'] = audio.shape[0]
                else:
                    result['channels'] = 1
            
            if verbose:
                print(f"📁 File: {audio_path.name}")
                print(f"   Sample Rate: {sr} Hz")
                print(f"   Duration: {info:.2f} seconds")
            
            if sr == 16000:
                result['is_16khz'] = True
                if verbose:
                    print("   ✅ Sample rate đúng (16kHz)")
            else:
                result['needs_resample'] = True
                if verbose:
                    print(f"   ⚠️  Sample rate KHÔNG đúng! Cần resample từ {sr}Hz → 16kHz")
                    print(f"   🚨 Nếu không resample, model sẽ nghe như 'người ngoài hành tinh'!")
                    
        except Exception as e2:
            print(f"❌ Không thể đọc file: {e2}")
            result['exists'] = False
            return result
    
    return result


def check_multiple_files(file_paths: list, show_summary: bool = True) -> dict:
    """
    Kiểm tra sample rate của nhiều file.
    
    Args:
        file_paths: Danh sách đường dẫn file
        show_summary: Hiển thị tổng kết
        
    Returns:
        Dictionary tổng kết
    """
    results = []
    summary = {
        'total': len(file_paths),
        'checked': 0,
        'is_16khz': 0,
        'needs_resample': 0,
        'errors': 0,
        'sample_rates': {}
    }
    
    print(f"🔍 Đang kiểm tra {len(file_paths)} file...")
    print("-" * 60)
    
    for i, file_path in enumerate(file_paths, 1):
        result = check_sample_rate(file_path, verbose=True)
        if result:
            results.append(result)
            summary['checked'] += 1
            
            if result['is_16khz']:
                summary['is_16khz'] += 1
            elif result['needs_resample']:
                summary['needs_resample'] += 1
            
            # Đếm sample rates
            sr = result.get('torchaudio_sr') or result.get('librosa_sr')
            if sr:
                summary['sample_rates'][sr] = summary['sample_rates'].get(sr, 0) + 1
        else:
            summary['errors'] += 1
        
        if i < len(file_paths):
            print()
    
    if show_summary:
        print("\n" + "=" * 60)
        print("📊 TỔNG KẾT")
        print("=" * 60)
        print(f"Tổng số file: {summary['total']}")
        print(f"Đã kiểm tra: {summary['checked']}")
        print(f"✅ Đúng 16kHz: {summary['is_16khz']} ({summary['is_16khz']/summary['checked']*100:.1f}%)" if summary['checked'] > 0 else "")
        print(f"⚠️  Cần resample: {summary['needs_resample']} ({summary['needs_resample']/summary['checked']*100:.1f}%)" if summary['checked'] > 0 else "")
        print(f"❌ Lỗi: {summary['errors']}")
        
        if summary['sample_rates']:
            print("\n📈 Phân bố Sample Rate:")
            for sr, count in sorted(summary['sample_rates'].items()):
                percentage = count / summary['checked'] * 100
                status = "✅" if sr == 16000 else "⚠️"
                print(f"   {status} {sr} Hz: {count} files ({percentage:.1f}%)")
    
    return {
        'results': results,
        'summary': summary
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Kiểm tra sample rate của file audio WAV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Kiểm tra 1 file
  python scripts/check_sample_rate.py data/audio/sample.wav
  
  # Kiểm tra nhiều file
  python scripts/check_sample_rate.py data/audio/*.wav
  
  # Kiểm tra toàn bộ dataset
  python scripts/check_sample_rate.py --dataset data/processed/full_merged_dataset/train
        """
    )
    
    parser.add_argument(
        'files',
        nargs='*',
        help='Đường dẫn đến file audio (có thể nhiều file)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Đường dẫn đến thư mục dataset (sẽ kiểm tra tất cả file .wav)'
    )
    
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Chỉ kiểm tra N file đầu tiên (để test nhanh)'
    )
    
    args = parser.parse_args()
    
    if args.dataset:
        # Kiểm tra toàn bộ dataset
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"❌ Thư mục không tồn tại: {dataset_path}")
            sys.exit(1)
        
        # Tìm tất cả file .wav trong thư mục audio
        audio_dir = dataset_path / 'audio'
        if not audio_dir.exists():
            print(f"❌ Không tìm thấy thư mục audio: {audio_dir}")
            sys.exit(1)
        
        wav_files = list(audio_dir.glob('*.wav'))
        
        if args.sample:
            wav_files = wav_files[:args.sample]
            print(f"📝 Chỉ kiểm tra {len(wav_files)} file đầu tiên (--sample={args.sample})")
        
        if not wav_files:
            print(f"❌ Không tìm thấy file .wav trong {audio_dir}")
            sys.exit(1)
        
        check_multiple_files([str(f) for f in wav_files])
        
    elif args.files:
        # Kiểm tra các file được chỉ định
        if len(args.files) == 1:
            check_sample_rate(args.files[0])
        else:
            check_multiple_files(args.files)
    else:
        parser.print_help()
        sys.exit(1)

