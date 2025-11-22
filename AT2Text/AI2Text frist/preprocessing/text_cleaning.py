"""
Text preprocessing and normalization for Vietnamese ASR.
Handles text cleaning, normalization, and tokenization for Vietnamese language.
"""

import re
import unicodedata
from typing import List, Optional, Dict
import string


class VietnameseTextNormalizer:
    """Text normalizer specifically designed for Vietnamese language."""
    
    def __init__(self, lowercase: bool = True, 
                 remove_punctuation: bool = True,
                 normalize_unicode: bool = True):
        """Initialize Vietnamese text normalizer.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            normalize_unicode: Normalize Unicode characters
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        
        # Vietnamese specific mappings
        self.number_map = {
            '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
            '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
        }
        
        # Common abbreviations in Vietnamese
        self.abbreviation_map = {
            'tp.': 'thành phố',
            'tphcm': 'thành phố hồ chí minh',
            'hà nội': 'hà nội',
            'đà nẵng': 'đà nẵng',
            'cn': 'công nghệ',
            'tt': 'trung tâm',
            'bv': 'bệnh viện',
            'dh': 'đại học',
            'gd': 'giáo dục',
            'nxb': 'nhà xuất bản',
        }
        
        # Vietnamese punctuation to keep for natural speech
        self.vietnamese_punctuation = '.,;:!?'
    
    def normalize_unicode_text(self, text: str) -> str:
        """Normalize Unicode characters to NFC form.
        
        Args:
            text: Input text
            
        Returns:
            normalized_text: Unicode normalized text
        """
        # NFC normalization (canonical composition)
        return unicodedata.normalize('NFC', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace.
        
        Args:
            text: Input text
            
        Returns:
            cleaned_text: Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common Vietnamese abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            expanded_text: Text with expanded abbreviations
        """
        text_lower = text.lower()
        for abbr, expansion in self.abbreviation_map.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text_lower = re.sub(pattern, expansion, text_lower)
        
        return text_lower if self.lowercase else text
    
    def convert_numbers_to_words(self, text: str) -> str:
        """Convert digits to Vietnamese words.
        
        Args:
            text: Input text
            
        Returns:
            converted_text: Text with numbers as words
        """
        def replace_number(match):
            number = match.group(0)
            # Convert each digit
            return ' '.join(self.number_map.get(d, d) for d in number)
        
        # Match sequences of digits
        text = re.sub(r'\d+', replace_number, text)
        return text
    
    def remove_special_characters(self, text: str, 
                                   keep_vietnamese: bool = True) -> str:
        """Remove special characters and punctuation.
        
        Args:
            text: Input text
            keep_vietnamese: Keep Vietnamese tone marks
            
        Returns:
            cleaned_text: Text without special characters
        """
        if keep_vietnamese:
            # Keep Vietnamese characters, spaces, and optionally punctuation
            if not self.remove_punctuation:
                pattern = r'[^a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ\s.,;:!?]'
            else:
                pattern = r'[^a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ\s]'
        else:
            pattern = r'[^a-zA-Z\s]'
        
        text = re.sub(pattern, '', text)
        return text
    
    def normalize(self, text: str) -> str:
        """Complete normalization pipeline.
        
        Args:
            text: Input text
            
        Returns:
            normalized_text: Fully normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = self.normalize_unicode_text(text)
        
        # Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # Convert numbers to words
        text = self.convert_numbers_to_words(text)
        
        # Remove special characters
        text = self.remove_special_characters(text)
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = self.remove_extra_whitespace(text)
        
        return text
    
    def clean_transcript(self, text: str, 
                        remove_filler_words: bool = True) -> str:
        """Clean transcript for training.
        
        Args:
            text: Input transcript
            remove_filler_words: Remove filler words like "ừm", "à"
            
        Returns:
            cleaned_transcript: Cleaned transcript
        """
        text = self.normalize(text)
        
        # Remove Vietnamese filler words if requested
        if remove_filler_words:
            filler_words = ['ừm', 'à', 'ờ', 'ừ', 'ể', 'thì', 'này']
            pattern = r'\b(' + '|'.join(filler_words) + r')\b'
            text = re.sub(pattern, '', text)
            text = self.remove_extra_whitespace(text)
        
        return text


class Tokenizer:
    """Character-level tokenizer for Vietnamese ASR."""
    
    def __init__(self, vocab: Optional[List[str]] = None):
        """Initialize tokenizer.
        
        Args:
            vocab: Optional predefined vocabulary
        """
        if vocab is None:
            # Default Vietnamese character vocabulary
            vocab = self._build_default_vocab()
        
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.blank_token = '<blank>'  # For CTC loss
        
        self.pad_token_id = self.char_to_idx.get(self.pad_token, 0)
        self.unk_token_id = self.char_to_idx.get(self.unk_token, 1)
        self.blank_token_id = self.char_to_idx.get(self.blank_token, 0)
    
    def _build_default_vocab(self) -> List[str]:
        """Build default Vietnamese character vocabulary."""
        # Special tokens
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>', '<blank>']
        
        # Vietnamese alphabet (lowercase)
        vietnamese_chars = [
            'a', 'à', 'á', 'ả', 'ã', 'ạ',
            'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
            'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ',
            'b', 'c', 'd', 'đ',
            'e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
            'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ',
            'g', 'h',
            'i', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
            'k', 'l', 'm', 'n',
            'o', 'ò', 'ó', 'ỏ', 'õ', 'ọ',
            'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
            'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ',
            'p', 'q', 'r', 's', 't',
            'u', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
            'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự',
            'v', 'x',
            'y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'
        ]
        
        # Space
        space = [' ']
        
        return special_tokens + vietnamese_chars + space
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            token_ids: List of token IDs
        """
        return [self.char_to_idx.get(char, self.unk_token_id) for char in text]
    
    def decode(self, token_ids: List[int], 
               skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            text: Decoded text
        """
        special_tokens = {'<pad>', '<unk>', '<sos>', '<eos>', '<blank>'}
        chars = []
        
        for idx in token_ids:
            char = self.idx_to_char.get(idx, self.unk_token)
            if skip_special_tokens and char in special_tokens:
                continue
            chars.append(char)
        
        return ''.join(chars)
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w', encoding='utf-8') as f:
            for char in self.vocab:
                f.write(f"{char}\n")
    
    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = [line.strip() for line in f]
        
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}


def prepare_text_for_training(text: str, 
                              normalizer: Optional[VietnameseTextNormalizer] = None,
                              tokenizer: Optional[Tokenizer] = None) -> Dict:
    """Complete text preprocessing pipeline for training.
    
    Args:
        text: Raw transcript text
        normalizer: Text normalizer instance
        tokenizer: Tokenizer instance
        
    Returns:
        result: Dictionary with processed text and tokens
    """
    if normalizer is None:
        normalizer = VietnameseTextNormalizer()
    
    if tokenizer is None:
        tokenizer = Tokenizer()
    
    # Normalize text
    normalized_text = normalizer.normalize(text)
    
    # Tokenize
    token_ids = tokenizer.encode(normalized_text)
    
    return {
        'original_text': text,
        'normalized_text': normalized_text,
        'token_ids': token_ids,
        'num_tokens': len(token_ids)
    }


if __name__ == "__main__":
    # Test text normalization
    normalizer = VietnameseTextNormalizer()
    test_text = "Xin chào, tôi là trợ lý AI. Số điện thoại: 0123456789."
    print(f"Original: {test_text}")
    print(f"Normalized: {normalizer.normalize(test_text)}")
    
    # Test tokenizer
    tokenizer = Tokenizer()
    print(f"\nVocabulary size: {len(tokenizer)}")
    
    test_sentence = "xin chào việt nam"
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: {test_sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

