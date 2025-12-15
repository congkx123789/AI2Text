import argparse, json, torch, torch.nn as nn, torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from pathlib import Path
import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
from .dataset import ASRDataset, ManifestDataset, collate_batch
from .model import CRNNCTC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_audio", default="data/raw/audio", help="Used if manifest is not provided")
    ap.add_argument("--train_text",  default="data/raw/text",  help="Used if manifest is not provided")
    ap.add_argument("--manifest", default=None, help="Path to manifest.csv (processed dataset)")
    ap.add_argument("--audio_root", default=None, help="Root to prepend to audio_path in manifest")
    ap.add_argument("--timestamps", default=None, help="timestamps.json for optional trimming")
    ap.add_argument("--trim_segments", action="store_true", help="Trim audio to min/max segment using timestamps.json")
    ap.add_argument("--vocab",       default="data/processed/vocab.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto", help="cuda|cpu|mps|auto")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="enable automatic mixed precision on GPU (uses bfloat16)")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)")
    ap.add_argument("--max_grad_norm", type=float, default=None, help="Gradient clipping max norm (None to disable)")
    ap.add_argument("--empty_cache", action="store_true", help="Empty CUDA cache after each batch to save memory")
    ap.add_argument("--log_interval", type=int, default=50, help="print progress every N batches")
    ap.add_argument("--out", default="data/results/asr_ctc.pt", help="Final model output path")
    ap.add_argument("--checkpoint_dir", default="data/results/checkpoints", help="Directory to save checkpoints")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--log_dir", default="runs/asr_ctc", help="TensorBoard log directory")
    args = ap.parse_args()

    def progress_bar(step, total, width=24):
        filled = int(width * step / total)
        return "|" * filled + "." * (width - filled)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda":
        cudnn.benchmark = True

    if args.manifest:
        ds = ManifestDataset(
            manifest_path=args.manifest,
            vocab_path=args.vocab,
            audio_root=args.audio_root,
            timestamps_path=args.timestamps,
            trim_to_segments=args.trim_segments,
        )
    else:
        ds = ASRDataset(args.train_audio, args.train_text, args.vocab)
    # Memory optimization: Enable pin_memory for GPU (good for Ryzen 9 + RTX 5060 Ti)
    # With 16GB VRAM, we can use pin_memory even with larger batches
    pin_memory = device.type == "cuda" and args.batch_size <= 128
    
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if args.num_workers > 0 else None,  # Prefetch batches to reduce GPU starvation
    )
    model = CRNNCTC(n_mels=80, vocab_size=len(ds.vocab)).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    # Use new torch.amp API (fixes FutureWarning)
    scaler = GradScaler('cuda', enabled=args.amp and device.type == "cuda")
    # Always use bf16 when AMP is enabled (no fp16)
    if args.amp and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            autocast_ctx = lambda: autocast('cuda', dtype=torch.bfloat16)
            print("Using bfloat16 (bf16) precision with AMP")
        else:
            print("Warning: bf16 not supported on this GPU, falling back to full precision (fp32)")
            autocast_ctx = nullcontext
    else:
        autocast_ctx = nullcontext

    # Setup checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler.is_enabled():
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Resuming from epoch {start_epoch}")

    # Memory optimization: Enable empty cache if requested
    if args.empty_cache and device.type == "cuda":
        print("Memory optimization: Emptying CUDA cache after each batch")
    
    # Gradient accumulation info
    if args.gradient_accumulation_steps > 1:
        print(f"Gradient accumulation: {args.gradient_accumulation_steps} steps (effective batch size: {args.batch_size * args.gradient_accumulation_steps})")
    
    # TensorBoard writer (optional)
    writer = None
    if SummaryWriter is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path(args.log_dir) / f"run_{timestamp}"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logging to {log_dir}")
    else:
        print("TensorBoard not available (SummaryWriter import failed). Install with: pip install tensorboard")

    global_step = 0
    for ep in range(start_epoch, args.epochs+1):
        model.train(); total = 0.0
        opt.zero_grad()  # Zero gradients at start of epoch
        
        for i, (X, Xlen, Y, Ylen) in enumerate(dl, 1):
            X, Xlen, Y, Ylen = X.to(device, non_blocking=True), Xlen.to(device), Y.to(device), Ylen.to(device)
            
            with autocast_ctx():
                logits, out_lens = model(X, Xlen)      # (T,B,V)
                log_probs = logits.log_softmax(dim=-1)
                loss = ctc(log_probs, Y, out_lens, Ylen)
                # Scale loss by accumulation steps
                loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation: only step optimizer every N steps
            if i % args.gradient_accumulation_steps == 0:
                # Gradient clipping if specified
                if args.max_grad_norm is not None:
                    if scaler.is_enabled():
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
            
            total += loss.item() * args.gradient_accumulation_steps  # Scale back for logging
            
            # Empty cache to free memory
            if args.empty_cache and device.type == "cuda":
                torch.cuda.empty_cache()
            
            if i % args.log_interval == 0:
                bar = progress_bar(i, len(dl))
                current_loss = loss.item() * args.gradient_accumulation_steps
                print(f"epoch {ep} [{bar}] {i}/{len(dl)} | loss {current_loss:.3f}")
                if writer is not None:
                    writer.add_scalar("Loss/Train_step", current_loss, global_step)
        avg_loss = total/len(dl)
        print(f"epoch {ep} | loss {avg_loss:.3f}")
        if writer is not None:
            writer.add_scalar("Loss/Train_epoch", avg_loss, ep)

        # Save checkpoint after each epoch
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{ep}.pt"
        checkpoint = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': avg_loss,
        }
        if scaler.is_enabled():
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved final model to {args.out}")

    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
