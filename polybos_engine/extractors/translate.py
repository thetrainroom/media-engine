"""Query translation for CLIP search.

Translates non-English search queries to English for better CLIP performance,
since CLIP models are primarily trained on English text.
"""

import logging
from typing import Optional

import httpx
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

# Languages supported for translation (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {"de", "fr", "es", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "ru", "ja", "zh-cn", "zh-tw", "ko"}


def detect_language(text: str) -> Optional[str]:
    """Detect the language of the given text.

    Args:
        text: Text to detect language of

    Returns:
        ISO 639-1 language code (e.g., 'en', 'de', 'fr') or None if detection fails
    """
    if not text or len(text.strip()) < 3:
        return None

    try:
        lang = detect(text)
        return lang
    except LangDetectException as e:
        logger.debug(f"Language detection failed: {e}")
        return None


def translate_to_english(text: str, source_lang: Optional[str] = None, ollama_base_url: str = "http://localhost:11434") -> str:
    """Translate text to English using Ollama.

    Args:
        text: Text to translate
        source_lang: Source language code (auto-detected if not provided)
        ollama_base_url: Base URL for Ollama API

    Returns:
        Translated text in English, or original text if translation fails
    """
    if not text:
        return text

    # Detect language if not provided
    if source_lang is None:
        source_lang = detect_language(text)

    # If already English or detection failed, return original
    if source_lang is None or source_lang == "en":
        return text

    # Try translation with Ollama
    try:
        prompt = f"""Translate the following text to English. Only output the translation, nothing else.

Text: {text}

English translation:"""

        response = httpx.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": "qwen2.5:3b",  # Small, fast model for translation
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for deterministic translation
                    "num_predict": 100,  # Short output expected
                },
            },
            timeout=30.0,
        )

        if response.status_code == 200:
            result = response.json()
            translated = result.get("response", "").strip()
            if translated:
                logger.info(f"Translated query from {source_lang}: '{text}' -> '{translated}'")
                return translated
    except httpx.TimeoutException:
        logger.warning("Translation timed out, using original text")
    except httpx.ConnectError:
        logger.debug("Ollama not available for translation")
    except Exception as e:
        logger.warning(f"Translation failed: {e}")

    return text


def translate_query_for_clip(text: str, enable_translation: bool = True, ollama_base_url: str = "http://localhost:11434") -> tuple[str, Optional[str], bool]:
    """Translate a CLIP search query to English if needed.

    Args:
        text: Search query text
        enable_translation: Whether to enable translation
        ollama_base_url: Base URL for Ollama API

    Returns:
        Tuple of (processed_text, detected_language, was_translated)
    """
    if not enable_translation:
        return text, None, False

    source_lang = detect_language(text)

    if source_lang is None or source_lang == "en":
        return text, source_lang, False

    translated = translate_to_english(text, source_lang, ollama_base_url)
    was_translated = translated != text

    return translated, source_lang, was_translated
