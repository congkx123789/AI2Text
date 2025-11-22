"""
Vietnamese phonetic helpers for "sound-exactly" biasing.

This module provides phonetic processing for Vietnamese, used for:
- Phon2Vec training (phonetic embeddings)
- Contextual biasing based on sound similarity
- OOV word handling

Provides:
  - strip_diacritics(text): remove Vietnamese diacritics safely
  - telex_encode_syllable(s): Telex-style tone markers (s, f, r, x, j, z)
  - vn_soundex(word): compact sound-like code for fast fuzzy lookup
  - phonetic_tokens(text): convert a sentence to sequence of phonetic tokens

Note: This is rule-based and lightweight; suitable as a backbone for Phon2Vec.
"""

from __future__ import annotations
import re
try:
    from unidecode import unidecode
except ImportError:
    # Fallback if unidecode not available
    def unidecode(text):
        return text

_TONE_MARKERS = {
    # tone name -> telex marker
    "sac": "s",     # sắc
    "huyen": "f",   # huyền
    "hoi": "r",     # hỏi
    "nga": "x",     # ngã
    "nang": "j",    # nặng
    "none": "z",    # không dấu
}

_DIACRITIC_TABLE = str.maketrans({
    "á":"a","à":"a","ả":"a","ã":"a","ạ":"a","ă":"a","ắ":"a","ằ":"a","ẳ":"a","ẵ":"a","ặ":"a","â":"a","ấ":"a","ầ":"a","ẩ":"a","ẫ":"a","ậ":"a",
    "é":"e","è":"e","ẻ":"e","ẽ":"e","ẹ":"e","ê":"e","ế":"e","ề":"e","ể":"e","ễ":"e","ệ":"e",
    "í":"i","ì":"i","ỉ":"i","ĩ":"i","ị":"i",
    "ó":"o","ò":"o","ỏ":"o","õ":"o","ọ":"o","ô":"o","ố":"o","ồ":"o","ổ":"o","ỗ":"o","ộ":"o","ơ":"o","ớ":"o","ờ":"o","ở":"o","ỡ":"o","ợ":"o",
    "ú":"u","ù":"u","ủ":"u","ũ":"u","ụ":"u","ư":"u","ứ":"u","ừ":"u","ử":"u","ữ":"u","ự":"u",
    "ý":"y","ỳ":"y","ỷ":"y","ỹ":"y","ỵ":"y",
    "đ":"d",
    "Á":"A","À":"A","Ả":"A","Ã":"A","Ạ":"A","Ă":"A","Ắ":"A","Ằ":"A","Ẳ":"A","Ẵ":"A","Ặ":"A","Â":"A","Ấ":"A","Ầ":"A","Ẩ":"A","Ẫ":"A","Ậ":"A",
    "É":"E","È":"E","Ẻ":"E","Ẽ":"E","Ẹ":"E","Ê":"E","Ế":"E","Ề":"E","Ể":"E","Ễ":"E","Ệ":"E",
    "Í":"I","Ì":"I","Ỉ":"I","Ĩ":"I","Ị":"I",
    "Ó":"O","Ò":"O","Ỏ":"O","Õ":"O","Ọ":"O","Ô":"O","Ố":"O","Ồ":"O","Ổ":"O","Ỗ":"O","Ộ":"O","Ơ":"O","Ớ":"O","Ờ":"O","Ở":"O","Ỡ":"O","Ợ":"O",
    "Ú":"U","Ù":"U","Ủ":"U","Ũ":"U","Ụ":"U","Ư":"U","Ứ":"U","Ừ":"U","Ử":"U","Ữ":"U","Ự":"U",
    "Ý":"Y","Ỳ":"Y","Ỷ":"Y","Ỹ":"Y","Ỵ":"Y",
    "Đ":"D"
})

def strip_diacritics(text: str) -> str:
    """
    Remove Vietnamese diacritics safely.
    
    Args:
        text: Input Vietnamese text with diacritics
        
    Returns:
        text without diacritics
    """
    # Use explicit map for Vietnamese + fallback unidecode for other symbols
    result = text.translate(_DIACRITIC_TABLE)
    # Fallback for characters not in map
    if result != text:
        return result
    return unidecode(text)

_NON_ALPHA = re.compile(r"[^a-zA-Z0-9]+")

def simple_tokenize(text: str) -> list[str]:
    """
    Simple tokenization - split by whitespace.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return [t for t in re.split(r"\s+", text) if t]

def detect_tone(syllable: str) -> str:
    """
    Detect Vietnamese tone from syllable.
    
    Args:
        syllable: Vietnamese syllable with diacritics
        
    Returns:
        Tone name: "sac", "huyen", "hoi", "nga", "nang", or "none"
    """
    orig = syllable
    if any(c in orig for c in "áắấéếíóốớúứý"):
        return "sac"
    if any(c in orig for c in "àằầèềìòồờùừỳ"):
        return "huyen"
    if any(c in orig for c in "ảẳẩẻểỉỏổởủửỷ"):
        return "hoi"
    if any(c in orig for c in "ãẵẫẽễĩõỗỡũữỹ"):
        return "nga"
    if any(c in orig for c in "ạặậẹệịọộợụựỵ"):
        return "nang"
    return "none"

def telex_encode_syllable(syllable: str, with_tone: bool = True) -> str:
    """
    Encode syllable in Telex format with tone marker.
    
    Args:
        syllable: Vietnamese syllable
        with_tone: Whether to append tone marker
        
    Returns:
        Telex-encoded syllable (e.g., "viet" + "s" for "việt" with sắc tone)
    """
    tone = detect_tone(syllable)
    base = strip_diacritics(syllable).lower()
    base = _NON_ALPHA.sub("", base)
    if not base:
        return ""
    if with_tone:
        marker = _TONE_MARKERS.get(tone, "z")
        return f"{base}{marker}"
    return base

def vn_soundex(word: str) -> str:
    """
    Very compact sound-like hash for Vietnamese.
    
    Similar to Soundex but optimized for Vietnamese phonetics:
    - Strip diacritics
    - Collapse double consonants
    - Map end-consonants to coarse classes
    
    Args:
        word: Vietnamese word
        
    Returns:
        Soundex code for the word
    """
    w = strip_diacritics(word).lower()
    w = _NON_ALPHA.sub("", w)
    if not w:
        return ""
    # Collapse common consonant clusters
    for src, dst in [("ch","c"),("tr","c"),("ph","f"),("th","t"),("ng","q"),
                     ("kh","k"),("gh","g"),("gi","z"),("qu","w")]:
        w = w.replace(src, dst)
    # Map finals
    if w.endswith(("c","t","p")):
        w = w[:-1] + "k"
    return w

def phonetic_tokens(text: str, telex: bool = True, tone_token: bool = True) -> list[str]:
    """
    Convert text to sequence of phonetic tokens.
    
    This function converts Vietnamese text into phonetic tokens that can be
    used for Phon2Vec training or sound-based similarity matching.
    
    Args:
        text: Vietnamese text
        telex: Use Telex encoding (True) or Soundex (False)
        tone_token: Append tone marker to tokens
        
    Returns:
        List of phonetic tokens
        
    Example:
        >>> phonetic_tokens("xin chào việt nam", telex=True, tone_token=True)
        ['xinz', 'chaof', 'vietj', 'namz']
    """
    toks = simple_tokenize(text)
    out = []
    for tok in toks:
        if telex:
            code = telex_encode_syllable(tok, with_tone=tone_token)
        else:
            code = vn_soundex(tok)
        if code:
            out.append(code)
    return out


# Export main function
__all__ = ['strip_diacritics', 'telex_encode_syllable', 'vn_soundex', 'phonetic_tokens']

