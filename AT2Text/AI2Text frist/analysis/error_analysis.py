"""
Error analysis tools for ASR predictions.

Provides detailed analysis of ASR errors:
- WER/CER breakdown by word frequency
- Confusion matrices
- Error type analysis (insertions, deletions, substitutions)
"""

from typing import List, Dict, Tuple, Counter
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import calculate_wer, calculate_cer, levenshtein_distance


class ErrorAnalyzer:
    """
    Error analyzer for ASR predictions.
    """
    
    def __init__(self):
        """Initialize error analyzer."""
        self.references = []
        self.predictions = []
        self.word_counts = Counter()
    
    def add_predictions(self, references: List[str], predictions: List[str]):
        """
        Add predictions for analysis.
        
        Args:
            references: Ground truth texts
            predictions: Predicted texts
        """
        self.references.extend(references)
        self.predictions.extend(predictions)
        
        # Count word frequencies in references
        for ref in references:
            words = ref.lower().split()
            self.word_counts.update(words)
    
    def analyze_by_frequency(self, bins: List[int] = [1, 10, 100, float('inf')]) -> Dict:
        """
        Analyze WER by word frequency bins.
        
        Args:
            bins: Frequency bins [min1, max1, min2, max2, ...]
            
        Returns:
            Dictionary with WER/CER per frequency bin
        """
        bins = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        
        results = {}
        
        for bin_min, bin_max in bins:
            bin_refs = []
            bin_preds = []
            
            for ref, pred in zip(self.references, self.predictions):
                ref_words = ref.lower().split()
                pred_words = pred.lower().split()
                
                # Check if reference has words in this frequency bin
                has_words_in_bin = False
                for word in ref_words:
                    freq = self.word_counts.get(word, 0)
                    if bin_min <= freq < bin_max:
                        has_words_in_bin = True
                        break
                
                if has_words_in_bin:
                    bin_refs.append(ref)
                    bin_preds.append(pred)
            
            if bin_refs:
                wer = calculate_wer(bin_refs, bin_preds)
                cer = calculate_cer(bin_refs, bin_preds)
                
                results[f"freq_{bin_min}_{bin_max}"] = {
                    'count': len(bin_refs),
                    'wer': wer,
                    'cer': cer
                }
        
        return results
    
    def confusion_matrix(self, level: str = 'character') -> pd.DataFrame:
        """
        Generate confusion matrix.
        
        Args:
            level: 'character' or 'word'
            
        Returns:
            DataFrame with confusion matrix
        """
        all_chars = set()
        
        # Collect all characters/words
        for ref, pred in zip(self.references, self.predictions):
            if level == 'character':
                ref_tokens = list(ref.lower())
                pred_tokens = list(pred.lower())
            else:
                ref_tokens = ref.lower().split()
                pred_tokens = pred.lower().split()
            
            all_chars.update(ref_tokens)
            all_chars.update(pred_tokens)
        
        all_chars = sorted(list(all_chars))
        char_to_idx = {char: i for i, char in enumerate(all_chars)}
        
        # Build confusion matrix
        matrix = np.zeros((len(all_chars), len(all_chars)))
        
        for ref, pred in zip(self.references, self.predictions):
            if level == 'character':
                ref_tokens = list(ref.lower())
                pred_tokens = list(pred.lower())
            else:
                ref_tokens = ref.lower().split()
                pred_tokens = pred.lower().split()
            
            # Align tokens (simplified - uses Levenshtein alignment)
            # In practice, use edit distance alignment
            min_len = min(len(ref_tokens), len(pred_tokens))
            for i in range(min_len):
                ref_char = ref_tokens[i] if i < len(ref_tokens) else '<PAD>'
                pred_char = pred_tokens[i] if i < len(pred_tokens) else '<PAD>'
                
                if ref_char in char_to_idx and pred_char in char_to_idx:
                    matrix[char_to_idx[ref_char], char_to_idx[pred_char]] += 1
        
        return pd.DataFrame(matrix, index=all_chars, columns=all_chars)
    
    def error_type_analysis(self) -> Dict:
        """
        Analyze error types: insertions, deletions, substitutions.
        
        Returns:
            Dictionary with error counts and rates
        """
        total_insertions = 0
        total_deletions = 0
        total_substitutions = 0
        total_ref_words = 0
        total_pred_words = 0
        
        for ref, pred in zip(self.references, self.predictions):
            ref_words = ref.lower().split()
            pred_words = pred.lower().split()
            
            total_ref_words += len(ref_words)
            total_pred_words += len(pred_words)
            
            # Calculate edit distance
            dist = levenshtein_distance(ref_words, pred_words)
            # Note: For detailed operation tracking, would need edit alignment algorithm
            
            # Simplified error counting (approximate)
            # Proper operation tracking requires edit alignment
            if len(pred_words) > len(ref_words):
                total_insertions += len(pred_words) - len(ref_words)
            elif len(pred_words) < len(ref_words):
                total_deletions += len(ref_words) - len(pred_words)
            # Substitutions approximated from edit distance
            total_substitutions += max(0, dist - abs(len(ref_words) - len(pred_words)))
        
        total_errors = total_insertions + total_deletions + total_substitutions
        
        return {
            'total_insertions': total_insertions,
            'total_deletions': total_deletions,
            'total_substitutions': total_substitutions,
            'total_errors': total_errors,
            'total_reference_words': total_ref_words,
            'total_predicted_words': total_pred_words,
            'insertion_rate': total_insertions / total_ref_words if total_ref_words > 0 else 0,
            'deletion_rate': total_deletions / total_ref_words if total_ref_words > 0 else 0,
            'substitution_rate': total_substitutions / total_ref_words if total_ref_words > 0 else 0
        }
    
    def generate_report(self, output_path: str):
        """
        Generate comprehensive error analysis report.
        
        Args:
            output_path: Path to save report
        """
        report = []
        report.append("="*70)
        report.append("ASR Error Analysis Report")
        report.append("="*70)
        report.append(f"Total samples: {len(self.references)}")
        report.append("")
        
        # Overall metrics
        overall_wer = calculate_wer(self.references, self.predictions)
        overall_cer = calculate_cer(self.references, self.predictions)
        
        report.append("Overall Metrics:")
        report.append(f"  WER: {overall_wer:.4f}")
        report.append(f"  CER: {overall_cer:.4f}")
        report.append("")
        
        # Frequency analysis
        report.append("WER by Word Frequency:")
        freq_analysis = self.analyze_by_frequency()
        for bin_name, metrics in freq_analysis.items():
            report.append(f"  {bin_name}: WER={metrics['wer']:.4f}, "
                         f"CER={metrics['cer']:.4f}, "
                         f"Count={metrics['count']}")
        report.append("")
        
        # Error type analysis
        report.append("Error Type Analysis:")
        error_types = self.error_type_analysis()
        report.append(f"  Insertions: {error_types['total_insertions']} "
                     f"({error_types['insertion_rate']:.4f})")
        report.append(f"  Deletions: {error_types['total_deletions']} "
                     f"({error_types['deletion_rate']:.4f})")
        report.append(f"  Substitutions: {error_types['total_substitutions']} "
                     f"({error_types['substitution_rate']:.4f})")
        
        # Write report
        report_text = "\n".join(report)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"[OK] Report saved to {output_path}")
        print("\n" + report_text)


if __name__ == "__main__":
    # Test error analyzer
    analyzer = ErrorAnalyzer()
    
    references = [
        "xin chào việt nam",
        "tôi là sinh viên",
        "hôm nay trời đẹp"
    ]
    
    predictions = [
        "xin chào việt nam",
        "tôi là sinh viên học",
        "hôm nay trời đẹp quá"
    ]
    
    analyzer.add_predictions(references, predictions)
    
    # Generate report
    analyzer.generate_report("analysis/error_report.txt")
    
    # Confusion matrix
    cm = analyzer.confusion_matrix(level='word')
    print("\nConfusion Matrix (sample):")
    print(cm.head(10))

