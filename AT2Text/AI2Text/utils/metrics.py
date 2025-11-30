"""
Evaluation metrics for ASR.
Includes WER (Word Error Rate) and CER (Character Error Rate).
"""

from typing import List
import numpy as np


def levenshtein_distance(ref: List, hyp: List) -> int:
    """Calculate Levenshtein distance between two sequences.
    
    Args:
        ref: Reference sequence
        hyp: Hypothesis sequence
        
    Returns:
        distance: Edit distance
    """
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1   # Substitution
                )
    
    return dp[m][n]


def calculate_wer(references: List[str], hypotheses: List[str]) -> float:
    """Calculate Word Error Rate.
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        
    Returns:
        wer: Word error rate
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    total_words = 0
    total_distance = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        if len(ref_words) == 0:
            continue
        
        distance = levenshtein_distance(ref_words, hyp_words)
        total_distance += distance
        total_words += len(ref_words)
    
    if total_words == 0:
        return 0.0
    
    wer = total_distance / total_words
    return wer


def calculate_cer(references: List[str], hypotheses: List[str]) -> float:
    """Calculate Character Error Rate.
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        
    Returns:
        cer: Character error rate
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    total_chars = 0
    total_distance = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_chars = list(ref.replace(' ', ''))
        hyp_chars = list(hyp.replace(' ', ''))
        
        if len(ref_chars) == 0:
            continue
        
        distance = levenshtein_distance(ref_chars, hyp_chars)
        total_distance += distance
        total_chars += len(ref_chars)
    
    if total_chars == 0:
        return 0.0
    
    cer = total_distance / total_chars
    return cer


def calculate_accuracy(references: List[str], hypotheses: List[str]) -> float:
    """Calculate sentence-level accuracy.
    
    Args:
        references: List of reference transcripts
        hypotheses: List of hypothesis transcripts
        
    Returns:
        accuracy: Proportion of exact matches
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")
    
    exact_matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref == hyp)
    accuracy = exact_matches / len(references)
    
    return accuracy


if __name__ == "__main__":
    # Test metrics
    references = [
        "xin chào việt nam",
        "tôi là sinh viên",
        "hôm nay trời đẹp"
    ]
    
    hypotheses = [
        "xin chào việt nam",
        "tôi là học sinh",
        "hôm nay trời mưa"
    ]
    
    wer = calculate_wer(references, hypotheses)
    cer = calculate_cer(references, hypotheses)
    acc = calculate_accuracy(references, hypotheses)
    
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")
    print(f"Accuracy: {acc:.4f}")

