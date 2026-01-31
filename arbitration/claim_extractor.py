"""
Claim Extractor Module

Extracts atomic factual claims from LLM responses for arbitration analysis.
"""

import re
import logging
from typing import Set, List

logger = logging.getLogger(__name__)


class ClaimExtractor:
    """Extracts and normalizes factual claims from text responses."""
    
    # Sentence ending patterns
    SENTENCE_ENDINGS = r'[.!?]'
    
    # Patterns to identify non-factual content
    NON_FACTUAL_PATTERNS = [
        r'^\s*(?:however|moreover|furthermore|additionally|in conclusion)',
        r'^\s*(?:i think|i believe|in my opinion|perhaps|maybe)',
        r'^\s*(?:let me|let\'s|please|you should)',
    ]
    
    def __init__(self, min_claim_length: int = 15):
        """
        Initialize the claim extractor.
        
        Args:
            min_claim_length: Minimum character length for valid claims
        """
        self.min_claim_length = min_claim_length
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence strings
        """
        # Replace common abbreviations to avoid false splits
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.', r'\1<DOT>', text)
        
        # Split on sentence boundaries
        sentences = re.split(self.SENTENCE_ENDINGS, text)
        
        # Restore dots and clean up
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences]
        
        # Filter empty sentences
        return [s for s in sentences if s]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Determine if a sentence represents a factual claim.
        
        Args:
            sentence: Sentence to evaluate
            
        Returns:
            True if sentence is a factual claim
        """
        sentence_lower = sentence.lower()
        
        # Check for non-factual patterns
        for pattern in self.NON_FACTUAL_PATTERNS:
            if re.search(pattern, sentence_lower):
                return False
        
        # Check minimum length
        if len(sentence) < self.min_claim_length:
            return False
        
        # Must contain a verb (simple heuristic: look for common verb patterns)
        verb_indicators = ['is', 'are', 'was', 'were', 'has', 'have', 'had', 
                          'does', 'do', 'did', 'can', 'will', 'would', 'should',
                          'contains', 'includes', 'consists', 'involves', 'requires']
        
        if not any(verb in sentence_lower.split() for verb in verb_indicators):
            return False
        
        return True
    
    def _normalize_claim(self, claim: str) -> str:
        """
        Normalize a claim for comparison.
        
        Args:
            claim: Raw claim text
            
        Returns:
            Normalized claim string
        """
        # Convert to lowercase
        normalized = claim.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading/trailing punctuation
        normalized = normalized.strip('.,!?;: ')
        
        # Remove common filler words at boundaries
        normalized = re.sub(r'^\s*(?:well|so|now|then|also)\s+', '', normalized)
        
        return normalized
    
    def extract(self, text: str) -> Set[str]:
        """
        Extract atomic factual claims from text.
        
        Args:
            text: Input text from LLM response
            
        Returns:
            Set of normalized claim strings
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Empty or invalid text provided for claim extraction")
            return set()
        
        self.logger.info(f"Extracting claims from text: {len(text)} chars")
        
        try:
            # Split into sentences
            sentences = self._split_into_sentences(text)
            self.logger.debug(f"Split into {len(sentences)} sentences")
            
            # Filter and normalize claims
            claims = set()
            for sentence in sentences:
                if self._is_factual_claim(sentence):
                    normalized = self._normalize_claim(sentence)
                    if normalized:  # Ensure not empty after normalization
                        claims.add(normalized)
            
            self.logger.info(f"Extracted {len(claims)} factual claims")
            return claims
            
        except Exception as e:
            self.logger.error(f"Error extracting claims: {str(e)}")
            raise


def extract_claims(text: str) -> Set[str]:
    """
    Main entry point for claim extraction.
    
    Args:
        text: Text from LLM response
        
    Returns:
        Set of normalized factual claims
    """
    extractor = ClaimExtractor()
    return extractor.extract(text)


# ============================================================================
# Advanced Claim Similarity (Optional Enhancement)
# ============================================================================

def compute_claim_similarity(claim1: str, claim2: str) -> float:
    """
    Compute semantic similarity between two claims using word overlap.
    
    Args:
        claim1: First claim
        claim2: Second claim
        
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize into words
    words1 = set(re.findall(r'\b\w+\b', claim1.lower()))
    words2 = set(re.findall(r'\b\w+\b', claim2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def find_similar_claims(claims: Set[str], threshold: float = 0.6) -> List[tuple]:
    """
    Find pairs of similar claims above a threshold.
    
    Args:
        claims: Set of claims to compare
        threshold: Minimum similarity score
        
    Returns:
        List of (claim1, claim2, similarity) tuples
    """
    similar_pairs = []
    claims_list = list(claims)
    
    for i, claim1 in enumerate(claims_list):
        for claim2 in claims_list[i+1:]:
            similarity = compute_claim_similarity(claim1, claim2)
            if similarity >= threshold:
                similar_pairs.append((claim1, claim2, similarity))
    
    return sorted(similar_pairs, key=lambda x: x[2], reverse=True)
