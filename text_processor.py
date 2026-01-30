"""
Text processing module for the OCR system.

This module handles post-processing of extracted text including cleaning,
formatting, validation, and enhancement of OCR results.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union
import unicodedata
from dataclasses import dataclass
import string
from collections import Counter
import difflib

from utils import Timer

logger = logging.getLogger(__name__)

@dataclass
class ProcessedText:
    """Data class for processed text results."""
    original_text: str
    cleaned_text: str
    formatted_text: str
    corrections_made: List[str]
    confidence_adjustment: float
    word_count: int
    line_count: int
    character_count: int
    language_detected: Optional[str] = None
    readability_score: Optional[float] = None

class TextProcessingError(Exception):
    """Custom exception for text processing errors."""
    pass

class TextProcessor:
    """
    Advanced text processor for cleaning and enhancing OCR results.

    This class provides comprehensive text processing capabilities including
    noise removal, spelling correction, formatting, and quality assessment.
    """

    def __init__(self):
        """Initialize the text processor."""
        self.common_ocr_errors = self._load_common_ocr_errors()
        self.word_patterns = self._compile_word_patterns()

    def _load_common_ocr_errors(self) -> Dict[str, str]:
        """
        Load common OCR error patterns and their corrections.

        Returns:
            Dictionary mapping error patterns to corrections
        """
        return {
            # Common character substitutions
            'rn': 'm',
            'cl': 'd',
            'li': 'h',
            'vv': 'w',
            'nn': 'n',
            '0': 'O',  # Zero to O
            '1': 'l',  # One to l (context dependent)
            '5': 'S',  # Five to S
            '8': 'B',  # Eight to B

            # Common word-level errors
            'tlie': 'the',
            'liave': 'have',
            'witli': 'with',
            'tliis': 'this',
            'tliat': 'that',
            'wliich': 'which',
            'wlien': 'when',
            'wliere': 'where',
            'tliey': 'they',
            'tliere': 'there',
            'tlirough': 'through',
            'arid': 'and',
            'oi': 'of',
            'to': 'to',
            'lor': 'for',
            'ilie': 'the',
            'ol': 'of',
            'lrom': 'from',
            'wilh': 'with',
            'lhe': 'the',
            'nol': 'not',
            'bul': 'but',
            'lhat': 'that',
            'lhis': 'this',
            'lhey': 'they',
            'lhere': 'there',
            'lhen': 'then',
            'lhrough': 'through',
        }

    def _compile_word_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile regular expression patterns for text processing.

        Returns:
            Dictionary of compiled regex patterns
        """
        return {
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n\s*\n'),
            'leading_trailing_spaces': re.compile(r'^\s+|\s+$', re.MULTILINE),
            'special_chars': re.compile(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'time': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b'),
            'currency': re.compile(r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s?(?:USD|EUR|GBP|dollars?|euros?|pounds?)'),
            'isolated_chars': re.compile(r'\b[a-zA-Z]\b'),
            'repeated_chars': re.compile(r'(.)\1{3,}'),
            'word_boundaries': re.compile(r'\b\w+\b'),
        }

    def remove_noise_characters(self, text: str) -> Tuple[str, List[str]]:
        """
        Remove noise characters and artifacts from OCR text.

        Args:
            text: Input text string

        Returns:
            Tuple of (cleaned_text, list_of_corrections)
        """
        corrections = []
        cleaned = text

        # Remove or replace common OCR artifacts
        noise_patterns = {
            r'[|¦]': 'l',  # Vertical bars to l
            r'[°º]': 'o',  # Degree symbols to o
            r'["""]': '"',  # Smart quotes to regular quotes
            r"[''']": "'",  # Smart apostrophes to regular apostrophes
            r'[–—]': '-',  # Em/en dashes to hyphens
            r'[…]': '...',  # Ellipsis to three dots
            r'[®©™]': '',   # Remove trademark symbols
            r'[†‡§¶]': '',  # Remove special symbols
            r'[^\x00-\x7F]+': '',  # Remove non-ASCII characters (optional)
        }

        for pattern, replacement in noise_patterns.items():
            old_text = cleaned
            cleaned = re.sub(pattern, replacement, cleaned)
            if old_text != cleaned:
                corrections.append(f"Replaced noise pattern: {pattern}")

        # Remove excessive whitespace
        old_text = cleaned
        cleaned = self.word_patterns['multiple_spaces'].sub(' ', cleaned)
        cleaned = self.word_patterns['multiple_newlines'].sub('\n\n', cleaned)
        if old_text != cleaned:
            corrections.append("Normalized whitespace")

        # Remove isolated single characters (likely OCR errors)
        words = cleaned.split()
        filtered_words = []
        removed_chars = []

        for word in words:
            # Keep single characters if they're meaningful
            if len(word) == 1 and word.lower() in 'aio':
                filtered_words.append(word)
            elif len(word) == 1:
                removed_chars.append(word)
            else:
                filtered_words.append(word)

        if removed_chars:
            cleaned = ' '.join(filtered_words)
            corrections.append(f"Removed isolated characters: {', '.join(removed_chars)}")

        return cleaned, corrections

    def correct_common_errors(self, text: str) -> Tuple[str, List[str]]:
        """
        Correct common OCR errors using predefined patterns.

        Args:
            text: Input text string

        Returns:
            Tuple of (corrected_text, list_of_corrections)
        """
        corrections = []
        corrected = text

        # Word-level corrections
        words = corrected.split()
        corrected_words = []

        for word in words:
            # Clean word (remove punctuation for matching)
            clean_word = word.strip(string.punctuation).lower()

            if clean_word in self.common_ocr_errors:
                # Preserve original case and punctuation
                correction = self.common_ocr_errors[clean_word]
                if word[0].isupper():
                    correction = correction.capitalize()

                # Preserve punctuation
                punctuation = ''.join(c for c in word if c in string.punctuation)
                corrected_word = correction + punctuation

                corrected_words.append(corrected_word)
                corrections.append(f"'{word}' → '{corrected_word}'")
            else:
                corrected_words.append(word)

        corrected = ' '.join(corrected_words)

        # Character-level corrections within words
        for error, correction in self.common_ocr_errors.items():
            if len(error) <= 2:  # Only apply to character-level errors
                pattern = r'\b\w*' + re.escape(error) + r'\w*\b'
                matches = re.findall(pattern, corrected, re.IGNORECASE)

                for match in matches:
                    if error.lower() in match.lower():
                        corrected_match = match.replace(error, correction)
                        corrected = corrected.replace(match, corrected_match)
                        corrections.append(f"Character correction: '{match}' → '{corrected_match}'")

        return corrected, corrections

    def fix_spacing_issues(self, text: str) -> Tuple[str, List[str]]:
        """
        Fix spacing issues common in OCR text.

        Args:
            text: Input text string

        Returns:
            Tuple of (fixed_text, list_of_corrections)
        """
        corrections = []
        fixed = text

        # Fix missing spaces after punctuation
        old_text = fixed
        fixed = re.sub(r'([.!?])([A-Z])', r'\1 \2', fixed)
        if old_text != fixed:
            corrections.append("Added spaces after sentence endings")

        # Fix missing spaces after commas
        old_text = fixed
        fixed = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', fixed)
        if old_text != fixed:
            corrections.append("Added spaces after punctuation")

        # Fix spaces before punctuation
        old_text = fixed
        fixed = re.sub(r'\s+([.!?,:;])', r'\1', fixed)
        if old_text != fixed:
            corrections.append("Removed spaces before punctuation")

        # Fix concatenated words (heuristic approach)
        words = fixed.split()
        fixed_words = []

        for word in words:
            if len(word) > 15:  # Likely concatenated word
                # Try to split on capital letters
                split_word = re.sub(r'([a-z])([A-Z])', r'\1 \2', word)
                if split_word != word and len(split_word.split()) > 1:
                    fixed_words.extend(split_word.split())
                    corrections.append(f"Split concatenated word: '{word}' → '{split_word}'")
                else:
                    fixed_words.append(word)
            else:
                fixed_words.append(word)

        fixed = ' '.join(fixed_words)

        return fixed, corrections

    def validate_text_structure(self, text: str) -> Dict[str, any]:
        """
        Validate the structure and quality of the text.

        Args:
            text: Input text string

        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'statistics': {},
            'quality_score': 0
        }

        if not text.strip():
            validation['is_valid'] = False
            validation['issues'].append("Empty text")
            return validation

        # Calculate basic statistics
        words = text.split()
        lines = text.split('\n')

        validation['statistics'] = {
            'character_count': len(text),
            'word_count': len(words),
            'line_count': len([line for line in lines if line.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'alpha_ratio': sum(c.isalpha() for c in text) / len(text) if text else 0,
            'digit_ratio': sum(c.isdigit() for c in text) / len(text) if text else 0,
            'space_ratio': sum(c.isspace() for c in text) / len(text) if text else 0,
        }

        # Quality checks
        quality_score = 50  # Base score

        # Check alpha ratio (should be reasonable for text)
        alpha_ratio = validation['statistics']['alpha_ratio']
        if 0.5 <= alpha_ratio <= 0.9:
            quality_score += 20
        elif alpha_ratio < 0.3:
            validation['issues'].append("Very low alphabetic character ratio")
            quality_score -= 20

        # Check average word length
        avg_word_length = validation['statistics']['avg_word_length']
        if 3 <= avg_word_length <= 8:
            quality_score += 15
        elif avg_word_length < 2 or avg_word_length > 12:
            validation['issues'].append("Unusual average word length")
            quality_score -= 10

        # Check for excessive special characters
        special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
        special_char_ratio = special_char_count / len(text) if text else 0

        if special_char_ratio > 0.2:
            validation['issues'].append("High special character ratio")
            quality_score -= 15

        # Check for repeated characters (OCR artifacts)
        repeated_matches = self.word_patterns['repeated_chars'].findall(text)
        if repeated_matches:
            validation['issues'].append(f"Repeated character sequences found: {len(repeated_matches)}")
            quality_score -= 10

        # Check for reasonable sentence structure
        sentences = re.split(r'[.!?]+', text)
        valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]

        if len(valid_sentences) > 0:
            quality_score += 10
        else:
            validation['issues'].append("No clear sentence structure")

        validation['quality_score'] = max(0, min(100, quality_score))

        if validation['quality_score'] < 30:
            validation['is_valid'] = False

        return validation

    def format_text(self, text: str, format_type: str = 'standard') -> str:
        """
        Format text according to specified formatting rules.

        Args:
            text: Input text string
            format_type: Type of formatting to apply

        Returns:
            Formatted text string
        """
        formatted = text

        if format_type == 'standard':
            # Standard formatting
            formatted = self._apply_standard_formatting(formatted)
        elif format_type == 'preserve_structure':
            # Preserve original structure as much as possible
            formatted = self._apply_minimal_formatting(formatted)
        elif format_type == 'clean':
            # Aggressive cleaning and formatting
            formatted = self._apply_aggressive_formatting(formatted)

        return formatted

    def _apply_standard_formatting(self, text: str) -> str:
        """Apply standard text formatting."""
        # Normalize whitespace
        formatted = self.word_patterns['multiple_spaces'].sub(' ', text)
        formatted = self.word_patterns['multiple_newlines'].sub('\n\n', formatted)

        # Ensure proper capitalization after sentence endings
        formatted = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), formatted)

        # Capitalize first letter of text
        if formatted and formatted[0].islower():
            formatted = formatted[0].upper() + formatted[1:]

        return formatted.strip()

    def _apply_minimal_formatting(self, text: str) -> str:
        """Apply minimal formatting to preserve structure."""
        # Only normalize excessive whitespace
        formatted = re.sub(r' {3,}', '  ', text)  # Reduce excessive spaces but preserve some
        formatted = re.sub(r'\n{4,}', '\n\n\n', formatted)  # Reduce excessive newlines

        return formatted.strip()

    def _apply_aggressive_formatting(self, text: str) -> str:
        """Apply aggressive formatting and cleaning."""
        # Remove all extra whitespace
        formatted = self.word_patterns['multiple_spaces'].sub(' ', text)
        formatted = self.word_patterns['multiple_newlines'].sub('\n', formatted)

        # Remove leading/trailing spaces from each line
        lines = formatted.split('\n')
        formatted = '\n'.join(line.strip() for line in lines if line.strip())

        # Ensure proper sentence structure
        formatted = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), formatted)

        return formatted.strip()

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the text (basic implementation).

        Args:
            text: Input text string

        Returns:
            Detected language code or None
        """
        # Simple language detection based on character patterns
        # This is a basic implementation - for production use, consider using langdetect library

        if not text.strip():
            return None

        # Count character frequencies
        char_freq = Counter(text.lower())

        # English indicators
        english_indicators = ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but']
        english_score = sum(1 for word in english_indicators if word in text.lower())

        # Basic heuristic
        if english_score >= 2:
            return 'en'

        return 'unknown'

    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate a basic readability score for the text.

        Args:
            text: Input text string

        Returns:
            Readability score (0-100, higher is more readable)
        """
        if not text.strip():
            return 0

        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]

        if not words or not sentences:
            return 0

        # Calculate basic metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word.strip(string.punctuation)) for word in words) / len(words)

        # Simple readability formula (modified Flesch-like)
        readability = 100 - (avg_sentence_length * 1.5) - (avg_word_length * 2)

        return max(0, min(100, readability))

    def process_text(self, text: str,
                    apply_corrections: bool = True,
                    format_type: str = 'standard') -> ProcessedText:
        """
        Complete text processing pipeline.

        Args:
            text: Input text string
            apply_corrections: Whether to apply error corrections
            format_type: Type of formatting to apply

        Returns:
            ProcessedText object with all processing results
        """
        with Timer("Text processing"):
            original_text = text
            current_text = text
            all_corrections = []

            # Step 1: Remove noise characters
            current_text, noise_corrections = self.remove_noise_characters(current_text)
            all_corrections.extend(noise_corrections)

            # Step 2: Fix spacing issues
            current_text, spacing_corrections = self.fix_spacing_issues(current_text)
            all_corrections.extend(spacing_corrections)

            # Step 3: Apply error corrections (if enabled)
            if apply_corrections:
                current_text, error_corrections = self.correct_common_errors(current_text)
                all_corrections.extend(error_corrections)

            cleaned_text = current_text

            # Step 4: Format text
            formatted_text = self.format_text(cleaned_text, format_type)

            # Step 5: Validate and analyze
            validation = self.validate_text_structure(formatted_text)

            # Step 6: Additional analysis
            language = self.detect_language(formatted_text)
            readability = self.calculate_readability_score(formatted_text)

            # Calculate confidence adjustment based on processing
            confidence_adjustment = self._calculate_confidence_adjustment(
                original_text, formatted_text, validation, all_corrections
            )

            # Calculate final statistics
            words = formatted_text.split()
            lines = formatted_text.split('\n')

            result = ProcessedText(
                original_text=original_text,
                cleaned_text=cleaned_text,
                formatted_text=formatted_text,
                corrections_made=all_corrections,
                confidence_adjustment=confidence_adjustment,
                word_count=len(words),
                line_count=len([line for line in lines if line.strip()]),
                character_count=len(formatted_text),
                language_detected=language,
                readability_score=readability
            )

            logger.info(f"Text processing completed: {len(all_corrections)} corrections, "
                       f"confidence adjustment: {confidence_adjustment:+.1f}")

            return result

    def _calculate_confidence_adjustment(self, original: str, processed: str,
                                       validation: Dict[str, any],
                                       corrections: List[str]) -> float:
        """
        Calculate confidence adjustment based on processing results.

        Args:
            original: Original text
            processed: Processed text
            validation: Text validation results
            corrections: List of corrections made

        Returns:
            Confidence adjustment value (-50 to +20)
        """
        adjustment = 0

        # Positive adjustments for improvements
        if len(corrections) > 0:
            # Small bonus for successful corrections
            adjustment += min(len(corrections) * 0.5, 5)

        # Quality-based adjustments
        quality_score = validation.get('quality_score', 50)
        if quality_score >= 80:
            adjustment += 10
        elif quality_score >= 60:
            adjustment += 5
        elif quality_score < 30:
            adjustment -= 20

        # Penalty for excessive changes
        if len(original) > 0:
            change_ratio = abs(len(processed) - len(original)) / len(original)
            if change_ratio > 0.3:  # More than 30% change
                adjustment -= 15

        # Penalty for validation issues
        if not validation.get('is_valid', True):
            adjustment -= 10

        return max(-50, min(20, adjustment))

def process_ocr_text(text: str,
                    apply_corrections: bool = True,
                    format_type: str = 'standard') -> ProcessedText:
    """
    Convenience function for processing OCR text.

    Args:
        text: Input OCR text
        apply_corrections: Whether to apply error corrections
        format_type: Type of formatting to apply

    Returns:
        ProcessedText object
    """
    processor = TextProcessor()
    return processor.process_text(text, apply_corrections, format_type)