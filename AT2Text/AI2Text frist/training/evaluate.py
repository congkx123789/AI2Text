"""
Evaluation and inference scripts for ASR model.
"""

import torch
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.asr_base import ASRModel
from preprocessing.audio_processing import AudioProcessor
from preprocessing.text_cleaning import Tokenizer, VietnameseTextNormalizer
from database.db_utils import ASRDatabase
from training.dataset import ASRDataset, collate_fn
from torch.utils.data import DataLoader
from utils.metrics import calculate_wer, calculate_cer


class ASREvaluator:
    """Evaluator for ASR models."""
    
    def __init__(self, model_path: str, config: dict):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup components
        self.tokenizer = Tokenizer()
        self.audio_processor = AudioProcessor(
            sample_rate=config.get('sample_rate', 16000),
            n_mels=config.get('n_mels', 80)
        )
        self.normalizer = VietnameseTextNormalizer()
        
        # Load model
        self.model = ASRModel(
            input_dim=config.get('n_mels', 80),
            vocab_size=len(self.tokenizer),
            d_model=config.get('d_model', 256),
            num_encoder_layers=config.get('num_encoder_layers', 6),
            num_heads=config.get('num_heads', 4),
            d_ff=config.get('d_ff', 1024),
            dropout=0.0  # No dropout during inference
        )
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
        print(f"Running on device: {self.device}")
    
    def _ctc_decode(self, tokens: list) -> str:
        """CTC greedy decoding."""
        # Remove consecutive duplicates
        collapsed = []
        prev = None
        for token in tokens:
            if token != prev:
                collapsed.append(token)
                prev = token
        
        # Remove blank tokens
        filtered = [t for t in collapsed if t != self.tokenizer.blank_token_id]
        
        return self.tokenizer.decode(filtered)
    
    @torch.no_grad()
    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            transcription: Predicted transcription
        """
        # Load and process audio
        audio, sr = self.audio_processor.load_audio(audio_path)
        audio = self.audio_processor.trim_silence(audio)
        mel_spec = self.audio_processor.extract_mel_spectrogram(audio)
        
        # Prepare input
        features = torch.from_numpy(mel_spec.T).unsqueeze(0).float().to(self.device)
        lengths = torch.tensor([features.size(1)]).to(self.device)
        
        # Forward pass
        logits, _ = self.model(features, lengths)
        
        # Decode
        predictions = torch.argmax(logits, dim=-1)
        pred_tokens = predictions[0].cpu().tolist()
        transcription = self._ctc_decode(pred_tokens)
        
        return transcription
    
    @torch.no_grad()
    def evaluate_dataset(self, data_loader: DataLoader) -> dict:
        """Evaluate model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            results: Dictionary with evaluation metrics
        """
        all_predictions = []
        all_references = []
        
        for batch in tqdm(data_loader, desc='Evaluating'):
            audio_features = batch['audio_features'].to(self.device)
            audio_lengths = batch['audio_lengths'].to(self.device)
            text_lengths = batch['text_lengths']
            transcripts = batch['transcripts']
            
            # Forward pass
            logits, output_lengths = self.model(audio_features, audio_lengths)
            
            # Decode predictions
            predictions = torch.argmax(logits, dim=-1)
            
            for i in range(predictions.size(0)):
                pred_tokens = predictions[i, :output_lengths[i]].cpu().tolist()
                pred_text = self._ctc_decode(pred_tokens)
                
                all_predictions.append(pred_text)
                all_references.append(transcripts[i])
        
        # Calculate metrics
        wer = calculate_wer(all_references, all_predictions)
        cer = calculate_cer(all_references, all_predictions)
        
        results = {
            'wer': wer,
            'cer': cer,
            'num_samples': len(all_predictions),
            'predictions': all_predictions,
            'references': all_references
        }
        
        return results
    
    def save_results(self, results: dict, output_path: str):
        """Save evaluation results to file.
        
        Args:
            results: Results dictionary
            output_path: Path to save results
        """
        df = pd.DataFrame({
            'reference': results['references'],
            'prediction': results['predictions']
        })
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Save metrics summary
        summary_path = Path(output_path).parent / 'metrics_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"WER: {results['wer']:.4f}\n")
            f.write(f"CER: {results['cer']:.4f}\n")
            f.write(f"Number of samples: {results['num_samples']}\n")
        
        print(f"Results saved to {output_path}")
        print(f"Metrics summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASR model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to single audio file to transcribe')
    parser.add_argument('--output', type=str, default='results/evaluation.csv',
                       help='Path to save results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = ASREvaluator(args.checkpoint, config)
    
    if args.audio:
        # Transcribe single file
        transcription = evaluator.transcribe_file(args.audio)
        print(f"\nAudio: {args.audio}")
        print(f"Transcription: {transcription}")
    else:
        # Evaluate on dataset
        db = ASRDatabase(config.get('database_path', 'database/asr_training.db'))
        data_df = db.get_split_data(args.split, config.get('split_version', 'v1'))
        
        print(f"Evaluating on {len(data_df)} samples from {args.split} split")
        
        # Create dataset and loader
        dataset = ASRDataset(
            data_df=data_df,
            audio_processor=evaluator.audio_processor,
            tokenizer=evaluator.tokenizer,
            apply_augmentation=False
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            collate_fn=collate_fn
        )
        
        # Evaluate
        results = evaluator.evaluate_dataset(data_loader)
        
        print(f"\nEvaluation Results:")
        print(f"WER: {results['wer']:.4f}")
        print(f"CER: {results['cer']:.4f}")
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluator.save_results(results, str(output_path))


if __name__ == '__main__':
    main()

