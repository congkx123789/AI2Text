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
                 normalize_unicode: bool = True,
                 remove_filler_words: bool = True):
        """Initialize Vietnamese text normalizer.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            normalize_unicode: Normalize Unicode characters
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_unicode = normalize_unicode
        self.remove_filler_words = remove_filler_words
        
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
            'dr': 'tiến sĩ',
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
        text_processed = text.lower()
        for abbr, expansion in self.abbreviation_map.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text_processed = re.sub(pattern, expansion, text_processed)
        
        return text_processed
    
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
    
    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words from text."""
        filler_words = ['ừm', 'à', 'ờ', 'ừ', 'ể', 'thì', 'này']
        pattern = r'\b(' + '|'.join(filler_words) + r')\b'
        cleaned = re.sub(pattern, '', text)
        return self.remove_extra_whitespace(cleaned)
    
    def _normalize(self, text: str, remove_filler_words: bool) -> str:
        """Normalization pipeline with configurable filler removal."""
        if not text:
            return ""
        
        working = text
        
        if self.normalize_unicode:
            working = self.normalize_unicode_text(working)
        
        working = self.expand_abbreviations(working)
        working = self.convert_numbers_to_words(working)
        working = self.remove_special_characters(working)
        
        if self.lowercase:
            working = working.lower()
        
        working = self.remove_extra_whitespace(working)
        
        if remove_filler_words:
            working = self._remove_filler_words(working)
        
        return working
    
    def normalize(self, text: str) -> str:
        """Complete normalization pipeline.
        
        Args:
            text: Input text
            
        Returns:
            normalized_text: Fully normalized text
        """
        return self._normalize(text, self.remove_filler_words)
    
    def clean_transcript(self, text: str, 
                        remove_filler_words: bool = True) -> str:
        """Clean transcript for training.
        
        Args:
            text: Input transcript
            remove_filler_words: Remove filler words like "ừm", "à"
            
        Returns:
            cleaned_transcript: Cleaned transcript
        """
        return self._normalize(text, remove_filler_words)


class BilingualTextNormalizer:
    """Language-aware normalizer for Vietnamese + English."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

        # Vietnamese number map
        self.vi_number_map = {
            '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
            '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
        }
        # English number map (simplified)
        self.en_number_map = {
            '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
            '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
        }

    def normalize(self, text: str, lang: str = 'vi') -> str:
        """Normalize text with language awareness."""
        if not text:
            return ""

        text = str(text)

        # 1. Unicode normalization
        text = unicodedata.normalize('NFC', text)

        # 2. Lowercase
        if self.lowercase:
            text = text.lower()

        # 3. Language specific number conversion
        lang = (lang or 'vi').lower()
        if lang == 'vi':
            text = self._convert_numbers(text, self.vi_number_map)
            # Vietnamese-specific abbreviation expansion can be added here
        elif lang == 'en':
            text = self._convert_numbers(text, self.en_number_map)
            # English-specific abbreviation expansion can be added here

        # 4. Remove punctuation (keep spaces)
        if self.remove_punctuation:
            # Keep Vietnamese chars + English chars + digits + spaces
            pattern = r'[^a-z0-9àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ\s]'
            text = re.sub(pattern, '', text)

        # 5. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _convert_numbers(self, text: str, map_dict: Dict[str, str]) -> str:
        """Simple digit replacement using provided map."""
        return ''.join(map_dict.get(c, c) for c in text)


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
        """Build default bilingual (Vietnamese + English) character vocabulary."""
        # Special tokens
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>', '<blank>']
        
        # English alphabet (lowercase)
        english_chars = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        
        # Vietnamese alphabet (lowercase) - includes all diacritics
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
        
        # Combine: special tokens + English + Vietnamese + space
        # Remove duplicates (English 'a' vs Vietnamese 'a' - keep Vietnamese version with diacritics)
        all_chars = list(set(english_chars + vietnamese_chars))
        all_chars.sort()  # Sort for consistency
        
        return special_tokens + all_chars + space
    
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

