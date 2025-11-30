"""
DEBUG CHECKLIST #3: Test "Overfit 1 Batch"

Đây là cách sửa nhanh nhất. Script này sẽ:
1. Load đúng 1 file audio và 1 câu text
2. Train model lặp đi lặp lại 100 lần trên đúng 1 file đó
3. Kiểm tra:
   - Nếu Loss không về ~0 và WER không về 0% -> Code model/Tokenizer bị sai
   - Nếu Loss về 0 ngon lành -> Do dữ liệu thật quá nhiễu hoặc learning rate sai
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml
import argparse
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor, AudioAugmenter
from preprocessing.text_cleaning import Tokenizer, BilingualTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import ASRDataset
from utils.metrics import calculate_wer, calculate_cer
from utils.logger import setup_logger


def test_overfit_1batch(config_path: str, sample_idx: int = 0):
    """Test overfitting on a single batch.
    
    Args:
        config_path: Path to config file
        sample_idx: Index of sample to use (default: 0, first sample)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger('OverfitTest', 'overfit_test.log')
    logger.info("=" * 80)
    logger.info("🔍 DEBUG CHECKLIST #3: Test 'Overfit 1 Batch'")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Mục đích: Kiểm tra model/tokenizer có hoạt động đúng không")
    logger.info("  - Nếu Loss không về ~0: Code model/Tokenizer bị sai")
    logger.info("  - Nếu Loss về 0: Do dữ liệu thật quá nhiễu hoặc learning rate sai")
    logger.info("")
    
    # Initialize database
    db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
    
    # Load data (get first sample)
    train_df = db.get_split_data('train', config.get('split_version', 'v1'))
    
    if len(train_df) == 0:
        logger.error("❌ Không có dữ liệu training!")
        return
    
    # Find a valid sample (file exists)
    from pathlib import Path
    valid_sample_idx = None
    
    # Try starting from sample_idx, then search forward
    start_idx = sample_idx if sample_idx < len(train_df) else 0
    for idx in range(start_idx, len(train_df)):
        file_path = train_df.iloc[idx]['file_path']
        if Path(file_path).exists():
            valid_sample_idx = idx
            break
    
    # If not found, search from beginning
    if valid_sample_idx is None:
        for idx in range(len(train_df)):
            file_path = train_df.iloc[idx]['file_path']
            if Path(file_path).exists():
                valid_sample_idx = idx
                break
    
    if valid_sample_idx is None:
        logger.error("❌ Không tìm thấy file audio nào tồn tại trong dataset!")
        logger.error("   Kiểm tra lại đường dẫn file trong database.")
        return
    
    if valid_sample_idx != sample_idx:
        logger.warning(f"⚠️  Sample {sample_idx} không tồn tại, dùng sample {valid_sample_idx} thay thế")
    
    single_sample_df = train_df.iloc[[valid_sample_idx]].copy()
    
    logger.info(f"📝 Sử dụng sample {valid_sample_idx}:")
    logger.info(f"   File: {single_sample_df.iloc[0]['file_path']}")
    logger.info(f"   Text: '{single_sample_df.iloc[0]['transcript']}'")
    logger.info(f"   File exists: {Path(single_sample_df.iloc[0]['file_path']).exists()}")
    logger.info("")
    
    # Setup preprocessing
    audio_processor = AudioProcessor(
        sample_rate=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80)
    )
    
    # NO augmentation for overfit test (we want consistent data)
    augmenter = None
    
    # Setup tokenizer
    tokenizer_type = config.get('tokenizer_type', 'char')
    if tokenizer_type == 'bpe':
        from preprocessing.bpe_tokenizer import BPETokenizer
        bpe_path = config.get('bpe_vocab_path', 'models/bilingual_bpe.json')
        tokenizer = BPETokenizer()
        tokenizer.load(bpe_path)
    else:
        tokenizer = Tokenizer()
    
    normalizer = BilingualTextNormalizer()
    
    # Create dataset with single sample
    dataset = ASRDataset(
        data_df=single_sample_df,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        normalizer=normalizer,
        augmenter=augmenter,
        apply_augmentation=False,  # No augmentation for overfit test
        cache_in_ram=False
    )
    
    # Create dataloader with batch_size=1
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Single process for debugging
        pin_memory=False
    )
    
    # Get first batch with error handling
    try:
        batch = next(iter(dataloader))
    except FileNotFoundError as e:
        logger.error(f"❌ Không tìm thấy file audio: {e}")
        logger.error("   Vui lòng kiểm tra lại đường dẫn file trong database.")
        return
    except Exception as e:
        logger.error(f"❌ Lỗi khi load batch: {e}")
        logger.error("   Vui lòng kiểm tra lại dữ liệu và preprocessing.")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Device: {device}")
    logger.info("")
    
    audio_features = batch['audio_features'].to(device)
    audio_lengths = batch['audio_lengths'].to(device)
    text_tokens = batch['text_tokens'].to(device)
    text_lengths = batch['text_lengths'].to(device)
    
    # Log input info
    logger.info("📊 Input Information:")
    logger.info(f"   Audio features shape: {audio_features.shape}")
    logger.info(f"   Audio length: {audio_lengths[0].item()} frames")
    logger.info(f"   Text tokens shape: {text_tokens.shape}")
    logger.info(f"   Text length: {text_lengths[0].item()} tokens")
    logger.info(f"   Text (decoded): '{tokenizer.decode(text_tokens[0, :text_lengths[0]].cpu().tolist())}'")
    logger.info("")
    
    # Setup model
    model = ASRModel(
        input_dim=config.get('n_mels', 80),
        vocab_size=len(tokenizer),
        d_model=config.get('d_model', 256),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        dropout=config.get('dropout', 0.1)
    )
    model.to(device)
    
    logger.info(f"🤖 Model parameters: {model.get_num_trainable_params():,}")
    logger.info("")
    
    # Setup optimizer (use higher LR for overfit test)
    learning_rate = config.get('learning_rate', 1e-4)
    # For overfit test, we can use higher LR since we're training on 1 sample
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate * 10,  # 10x higher for faster overfitting
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # CTC Loss
    criterion = nn.CTCLoss(
        blank=tokenizer.blank_token_id,
        zero_infinity=True,
        reduction='mean'
    )
    
    logger.info("🚀 Bắt đầu training trên 1 sample (100 iterations)...")
    logger.info("")
    
    # Training loop (100 iterations on same sample)
    model.train()
    losses = []
    predictions_history = []
    
    for iteration in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        logits, output_lengths = model(audio_features, audio_lengths)
        
        # Check output_lengths >= text_lengths
        if output_lengths[0] < text_lengths[0]:
            logger.error(f"❌ Iteration {iteration}: output_length ({output_lengths[0].item()}) < text_length ({text_lengths[0].item()})")
            logger.error("   → CTC Loss sẽ không hoạt động đúng!")
            break
        
        # CTC loss
        logits_t = logits.transpose(0, 1)
        log_probs = torch.log_softmax(logits_t, dim=-1)
        loss = criterion(log_probs, text_tokens, output_lengths, text_lengths)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"❌ Iteration {iteration}: Loss is NaN/Inf!")
            break
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Decode prediction
        predictions = torch.argmax(logits, dim=-1)
        pred_tokens = predictions[0, :output_lengths[0]].cpu().tolist()
        
        # CTC decode
        def ctc_decode(tokens):
            collapsed = []
            prev = None
            for token in tokens:
                if token != prev:
                    collapsed.append(token)
                    prev = token
            filtered = [t for t in collapsed if t != tokenizer.blank_token_id]
            return tokenizer.decode(filtered)
        
        pred_text = ctc_decode(pred_tokens)
        ref_text = tokenizer.decode(text_tokens[0, :text_lengths[0]].cpu().tolist())
        
        loss_val = loss.item()
        losses.append(loss_val)
        predictions_history.append(pred_text)
        
        # Log every 10 iterations
        if (iteration + 1) % 10 == 0 or iteration == 0:
            wer = calculate_wer([ref_text], [pred_text])
            cer = calculate_cer([ref_text], [pred_text])
            
            logger.info(f"Iteration {iteration+1:3d}/100:")
            logger.info(f"  Loss: {loss_val:.6f}")
            logger.info(f"  WER:  {wer:.4f}")
            logger.info(f"  CER:  {cer:.4f}")
            logger.info(f"  Pred: '{pred_text}'")
            logger.info(f"  True: '{ref_text}'")
            logger.info("")
    
    # Final results
    logger.info("=" * 80)
    logger.info("📊 KẾT QUẢ OVERFIT TEST")
    logger.info("=" * 80)
    logger.info("")
    
    final_loss = losses[-1]
    final_pred = predictions_history[-1]
    final_ref = tokenizer.decode(text_tokens[0, :text_lengths[0]].cpu().tolist())
    final_wer = calculate_wer([final_ref], [final_pred])
    final_cer = calculate_cer([final_ref], [final_pred])
    
    logger.info(f"Final Loss: {final_loss:.6f}")
    logger.info(f"Final WER:  {final_wer:.4f}")
    logger.info(f"Final CER:  {final_cer:.4f}")
    logger.info(f"Final Pred: '{final_pred}'")
    logger.info(f"Final True: '{final_ref}'")
    logger.info("")
    
    # Diagnosis
    logger.info("=" * 80)
    logger.info("🔍 CHẨN ĐOÁN")
    logger.info("=" * 80)
    logger.info("")
    
    if final_loss < 0.01 and final_wer < 0.1:
        logger.info("✅ THÀNH CÔNG: Loss về ~0 và WER về ~0%")
        logger.info("   → Model và Tokenizer hoạt động ĐÚNG")
        logger.info("   → Vấn đề nằm ở:")
        logger.info("     - Dữ liệu thật quá nhiễu")
        logger.info("     - Learning rate sai (quá nhỏ hoặc quá lớn)")
        logger.info("     - Batch size quá lớn")
        logger.info("     - Data augmentation quá mạnh")
    elif final_loss < 0.1:
        logger.warning("⚠️  Loss giảm nhưng chưa về 0")
        logger.warning("   → Có thể do:")
        logger.warning("     - Learning rate cần điều chỉnh")
        logger.warning("     - Model architecture cần tinh chỉnh")
    else:
        logger.error("❌ THẤT BẠI: Loss không về ~0")
        logger.error("   → Code model/Tokenizer bị SAI")
        logger.error("   → Kiểm tra:")
        logger.error("     1. CTC Loss calculation")
        logger.error("     2. Tokenizer encoding/decoding")
        logger.error("     3. Model forward pass")
        logger.error("     4. output_lengths vs text_lengths")
    
    logger.info("")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test overfitting on 1 batch')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Index of sample to use (default: 0)')
    args = parser.parse_args()
    
    test_overfit_1batch(args.config, args.sample_idx)

