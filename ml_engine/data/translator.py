"""
Category Translator - Chinese to English translation for COCO categories.

This module provides a context manager for translating Chinese category names
to English before training. The translation model is automatically loaded
and unloaded to manage VRAM efficiently.

Usage:
    with CategoryTranslator(model_path) as translator:
        coco_data = translator.translate_categories(coco_data)
    # VRAM automatically freed here
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

import torch

logger = logging.getLogger(__name__)


def has_chinese_characters(text: str) -> bool:
    """
    Check if text contains Chinese (CJK) characters.
    
    Args:
        text: String to check
        
    Returns:
        True if any CJK character found
    """
    return any('\u4e00' <= c <= '\u9fff' for c in text)


def clean_translation(text: str) -> str:
    """
    Clean up translation output for use as text prompts.
    
    Removes trailing punctuation that translation models sometimes add.
    This is important because Grounding DINO uses periods as class separators
    (e.g., "cat . dog . bird"), so extra periods would break the prompt format.
    
    Args:
        text: Raw translation output
        
    Returns:
        Cleaned text without trailing punctuation
        
    Examples:
        >>> clean_translation("apple.")
        'apple'
        >>> clean_translation("red apple,")
        'red apple'
        >>> clean_translation("  bag ear  ")
        'bag ear'
    """
    # Strip whitespace
    text = text.strip()
    
    # Remove trailing punctuation (period, comma, semicolon, colon, etc.)
    # Keep stripping in case of multiple (e.g., "apple..")
    while text and text[-1] in '.,:;!?。，；：！？':
        text = text[:-1].strip()
    
    return text


class CategoryTranslator:
    """
    Context manager for translating COCO category names from Chinese to English.
    
    Automatically manages translation model lifecycle:
    - Loads model on __enter__
    - Translates categories
    - Unloads model and clears VRAM on __exit__
    
    This ensures VRAM is freed before training models are loaded.
    
    Example:
        >>> from ml_engine.data.translator import CategoryTranslator
        >>> 
        >>> with CategoryTranslator("data/models/pretrained/opus-mt-zh-en") as translator:
        ...     coco_data = translator.translate_categories(coco_data)
        >>> # Model unloaded, VRAM freed
        >>> 
        >>> # Now safe to load training models
        >>> trainer = TeacherTrainer(...)
    """
    
    def __init__(self, model_path: str):
        """
        Initialize translator with model path.
        
        Args:
            model_path: Path to local translation model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def __enter__(self) -> 'CategoryTranslator':
        """Load translation model to GPU."""
        self._load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unload model and clear VRAM."""
        self._unload_model()
        return False  # Don't suppress exceptions
    
    def _load_model(self) -> None:
        """Load translation model and tokenizer."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Translation model not found: {self.model_path}\n"
                f"Please download Helsinki-NLP/opus-mt-zh-en to this location."
            )
        
        logger.info("Loading translation model from: %s", self.model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("  Model loaded on: %s (%.1fM parameters)", self.device, param_count)
        
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated() / 1024**2
            logger.info("  VRAM allocated: %.1f MB", vram_mb)
    
    def _unload_model(self) -> None:
        """Unload model and free VRAM."""
        logger.info("Unloading translation model...")
        
        # Delete model and tokenizer
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_allocated() / 1024**2
            logger.info("  VRAM after cleanup: %.1f MB", vram_mb)
        
        logger.info("  Translation model unloaded")
    
    def translate(self, text: str) -> str:
        """
        Translate a single Chinese text to English.
        
        Args:
            text: Chinese text to translate
            
        Returns:
            English translation (cleaned of trailing punctuation)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use within 'with' context.")
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=128)
        raw = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_translation(raw)
    
    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of Chinese texts to English.
        
        More efficient than calling translate() repeatedly.
        
        Args:
            texts: List of Chinese texts to translate
            
        Returns:
            List of English translations (cleaned of trailing punctuation)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use within 'with' context.")
        
        if not texts:
            return []
        
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs, max_length=128)
        raw_translations = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        return [clean_translation(t) for t in raw_translations]
    
    def translate_categories(
        self, 
        coco_data: Dict[str, Any],
        keep_original: bool = True
    ) -> Dict[str, Any]:
        """
        Translate Chinese category names in COCO data to English.
        
        Modifies the COCO data in-place and returns it.
        
        Args:
            coco_data: COCO format dictionary with 'categories' key
            keep_original: If True, store original name in 'name_original' field
            
        Returns:
            Modified COCO data with translated category names
        """
        categories = coco_data.get('categories', [])
        
        if not categories:
            logger.warning("No categories found in COCO data")
            return coco_data
        
        # Find categories that need translation
        chinese_categories = []
        chinese_indices = []
        
        for idx, cat in enumerate(categories):
            name = cat.get('name', '')
            if has_chinese_characters(name):
                chinese_categories.append(name)
                chinese_indices.append(idx)
        
        if not chinese_categories:
            logger.info("No Chinese categories found - skipping translation")
            return coco_data
        
        logger.info("Translating %d Chinese categories...", len(chinese_categories))
        
        # Batch translate all Chinese categories
        translations = self.translate_batch(chinese_categories)
        
        # Update categories in-place
        for idx, (orig_idx, translation) in enumerate(zip(chinese_indices, translations)):
            original_name = categories[orig_idx]['name']
            
            if keep_original:
                categories[orig_idx]['name_original'] = original_name
            
            categories[orig_idx]['name'] = translation
            logger.info("  %s → %s", original_name, translation)
        
        logger.info("Category translation complete")
        return coco_data


def translate_coco_categories(
    coco_data: Dict[str, Any],
    model_path: str,
    keep_original: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to translate COCO categories.
    
    Handles the full lifecycle: load model, translate, unload, clear VRAM.
    
    Args:
        coco_data: COCO format dictionary
        model_path: Path to translation model
        keep_original: If True, preserve original names in 'name_original' field
        
    Returns:
        COCO data with translated category names
        
    Example:
        >>> coco_data = translate_coco_categories(
        ...     coco_data,
        ...     model_path="data/models/pretrained/opus-mt-zh-en"
        ... )
    """
    with CategoryTranslator(model_path) as translator:
        return translator.translate_categories(coco_data, keep_original=keep_original)

