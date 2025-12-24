"""
Language detection utilities for ASR model.
Supports multiple methods: model output tags, text-based detection, and audio-based detection.
"""
import re
from typing import Optional, Tuple, Dict

try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import langid
    LANGID_AVAILABLE = True
except ImportError:
    LANGID_AVAILABLE = False


# Language tag patterns
LANG_TAG_PATTERN = re.compile(r'<\|(vi|en|ei)\|>')
SUPPORTED_LANGUAGES = {'vi': 'Vietnamese', 'en': 'English', 'ei': 'English'}


def extract_language_tag(text: str) -> Optional[str]:
    """
    Extract language tag from model output.
    Returns language code (vi, en, ei) or None if not found.
    """
    match = LANG_TAG_PATTERN.search(text)
    if match:
        lang_code = match.group(1)
        # Normalize: ei -> en
        return 'en' if lang_code == 'ei' else lang_code
    return None


def detect_language_from_text(text: str, method: str = 'auto') -> Tuple[str, float]:
    """
    Detect language from transcribed text.
    
    Args:
        text: Transcribed text (may contain language tags)
        method: 'auto', 'langdetect', or 'langid'
    
    Returns:
        Tuple of (language_code, confidence)
    """
    # First, try to extract language tag
    lang_tag = extract_language_tag(text)
    if lang_tag:
        return (lang_tag, 1.0)
    
    # Remove language tags if present for text-based detection
    clean_text = LANG_TAG_PATTERN.sub('', text).strip()
    
    if not clean_text or len(clean_text) < 3:
        return ('unknown', 0.0)
    
    # Try langdetect first (more accurate for Vietnamese)
    if (method == 'auto' or method == 'langdetect') and LANGDETECT_AVAILABLE:
        try:
            detected = detect(clean_text)
            # Get confidence
            langs = detect_langs(clean_text)
            confidence = langs[0].prob if langs else 0.5
            
            # Normalize language codes
            if detected == 'vi':
                return ('vi', confidence)
            elif detected in ['en', 'ei']:
                return ('en', confidence)
        except LangDetectException:
            pass
    
    # Fallback to langid
    if (method == 'auto' or method == 'langid') and LANGID_AVAILABLE:
        try:
            lang_code, confidence = langid.classify(clean_text)
            if lang_code == 'vi':
                return ('vi', confidence)
            elif lang_code == 'en':
                return ('en', confidence)
        except Exception:
            pass
    
    # Heuristic fallback: check for Vietnamese characters
    vietnamese_chars = set('àáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ')
    if any(char in vietnamese_chars for char in clean_text):
        # Count Vietnamese characters
        vi_char_count = sum(1 for char in clean_text if char in vietnamese_chars)
        total_chars = len([c for c in clean_text if c.isalpha()])
        if total_chars > 0 and vi_char_count / total_chars > 0.1:
            return ('vi', 0.7)
    
    # Default to English if no Vietnamese characters
    return ('en', 0.5)


def detect_language_from_audio_features(features, model=None) -> Tuple[str, float]:
    """
    Detect language from audio features using a language detection model.
    This is a placeholder for future implementation.
    
    Args:
        features: Audio features (log mel spectrogram)
        model: Optional language detection model
    
    Returns:
        Tuple of (language_code, confidence)
    """
    # TODO: Implement audio-based language detection
    # This would require training a separate language classifier
    return ('unknown', 0.0)


def get_language_info(text: str, method: str = 'auto') -> Dict[str, any]:
    """
    Get comprehensive language information from text.
    
    Returns:
        Dictionary with language_code, language_name, confidence, source
    """
    # Try tag extraction first
    lang_tag = extract_language_tag(text)
    if lang_tag:
        return {
            'language_code': lang_tag,
            'language_name': SUPPORTED_LANGUAGES.get(lang_tag, 'Unknown'),
            'confidence': 1.0,
            'source': 'model_tag',
            'method': 'tag_extraction'
        }
    
    # Text-based detection
    lang_code, confidence = detect_language_from_text(text, method)
    
    detection_method = 'unknown'
    if LANGDETECT_AVAILABLE and method in ['auto', 'langdetect']:
        detection_method = 'langdetect'
    elif LANGID_AVAILABLE and method in ['auto', 'langid']:
        detection_method = 'langid'
    elif lang_code != 'unknown':
        detection_method = 'heuristic'
    
    return {
        'language_code': lang_code,
        'language_name': SUPPORTED_LANGUAGES.get(lang_code, 'Unknown'),
        'confidence': confidence,
        'source': 'text_detection',
        'method': detection_method
    }


def clean_text_with_language(text: str, remove_tag: bool = True) -> Tuple[str, Optional[str]]:
    """
    Clean text and extract language information.
    
    Args:
        text: Text with optional language tag
        remove_tag: Whether to remove language tag from text
    
    Returns:
        Tuple of (cleaned_text, language_code)
    """
    lang_tag = extract_language_tag(text)
    
    if remove_tag:
        clean_text = LANG_TAG_PATTERN.sub('', text).strip()
    else:
        clean_text = text.strip()
    
    return (clean_text, lang_tag)

