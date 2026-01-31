"""
Confidence Scorer Module

Computes confidence scores for arbitrated LLM responses.
"""

import logging
from typing import Set, Dict, List
import math

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Computes confidence scores based on claim agreement patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compute_basic_confidence(
        self, 
        supported_claims: Set[str], 
        total_claims: int
    ) -> float:
        """
        Compute basic confidence score as ratio of supported to total claims.
        
        Args:
            supported_claims: Set of claims with consensus support
            total_claims: Total number of unique claims across all LLMs
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if total_claims == 0:
            self.logger.warning("Total claims is zero, returning 0.0 confidence")
            return 0.0
        
        num_supported = len(supported_claims)
        confidence = num_supported / total_claims
        
        self.logger.debug(
            f"Basic confidence: {num_supported}/{total_claims} = {confidence:.3f}"
        )
        
        return min(1.0, max(0.0, confidence))
    
    def compute_weighted_confidence(self, unanimous, supported, disputed, total):
        """
        Compute confidence with soft penalty for disputed claims.

        unanimous, supported, disputed are SETS of claims
        """
        # Convert sets to counts
        unanimous_count = len(unanimous)
        supported_count = len(supported)
        disputed_count = len(disputed)

        core_agreement = unanimous_count + supported_count

        if core_agreement == 0:
            return 0.0

        confidence = core_agreement / (core_agreement + 0.5 * disputed_count)
        return round(confidence, 3)


    
    
    def compute_entropy_based_confidence(
        self,
        agreement_table: Dict[str, List[str]],
        num_llms: int
    ) -> float:
        """
        Compute confidence using information entropy of agreement distribution.
        
        Lower entropy (more agreement) = higher confidence
        
        Args:
            agreement_table: Mapping of claims to supporting LLMs
            num_llms: Total number of LLMs
            
        Returns:
            Entropy-based confidence score between 0.0 and 1.0
        """
        if not agreement_table or num_llms == 0:
            return 0.0
        
        # Count distribution of agreement levels
        agreement_counts = {}
        for supporters in agreement_table.values():
            level = len(supporters)
            agreement_counts[level] = agreement_counts.get(level, 0) + 1
        
        total_claims = len(agreement_table)
        
        # Calculate entropy
        entropy = 0.0
        for count in agreement_counts.values():
            probability = count / total_claims
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Max entropy occurs when claims are evenly distributed across all agreement levels
        max_entropy = math.log2(num_llms + 1) if num_llms > 0 else 1.0
        
        # Normalize entropy to [0, 1], then invert (lower entropy = higher confidence)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        confidence = 1.0 - normalized_entropy
        
        self.logger.debug(
            f"Entropy-based confidence: {confidence:.3f} "
            f"(entropy: {entropy:.3f}, max_entropy: {max_entropy:.3f})"
        )
        
        return confidence
    
    def compute_composite_confidence(
        self,
        supported_claims: Set[str],
        disputed_claims: Set[str],
        unanimous_claims: Set[str],
        agreement_table: Dict[str, List[str]],
        total_claims: int,
        num_llms: int
    ) -> Dict[str, float]:
        """
        Compute multiple confidence metrics and return a composite score.
        
        Args:
            supported_claims: Claims with consensus support
            disputed_claims: Claims supported by only one LLM
            unanimous_claims: Claims supported by all LLMs
            agreement_table: Claim to supporters mapping
            total_claims: Total unique claims
            num_llms: Total number of LLMs
            
        Returns:
            Dictionary with individual and composite confidence scores
        """
        basic = self.compute_basic_confidence(supported_claims, total_claims)
        weighted = self.compute_weighted_confidence(
            supported_claims, disputed_claims, unanimous_claims, total_claims
        )
        entropy = self.compute_entropy_based_confidence(agreement_table, num_llms)
        
        # Composite score: weighted average of all metrics
        composite = (0.3 * basic + 0.5 * weighted + 0.2 * entropy)
        
        scores = {
            'basic': round(basic, 3),
            'weighted': round(weighted, 3),
            'entropy': round(entropy, 3),
            'composite': round(composite, 3)
        }
        
        self.logger.info(f"Composite confidence scores: {scores}")
        return scores


def compute_confidence(supported_claims: Set[str], total_claims: int) -> float:
    """
    Main entry point for basic confidence computation.
    
    Args:
        supported_claims: Set of claims with consensus support
        total_claims: Total number of unique claims
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    scorer = ConfidenceScorer()
    return scorer.compute_basic_confidence(supported_claims, total_claims)


def compute_advanced_confidence(
    supported_claims: Set[str],
    disputed_claims: Set[str],
    unanimous_claims: Set[str],
    total_claims: int
) -> float:
    """
    Compute weighted confidence score.
    
    Args:
        supported_claims: Claims with consensus support
        disputed_claims: Claims supported by only one LLM
        unanimous_claims: Claims supported by all LLMs
        total_claims: Total unique claims
        
    Returns:
        Weighted confidence score between 0.0 and 1.0
    """
    scorer = ConfidenceScorer()
    return scorer.compute_weighted_confidence(
        supported_claims, disputed_claims, unanimous_claims, total_claims
    )


# ============================================================================
# Confidence Interpretation
# ============================================================================

def interpret_confidence(confidence: float) -> str:
    """
    Provide human-readable interpretation of confidence score.
    
    Args:
        confidence: Confidence score between 0.0 and 1.0
        
    Returns:
        String interpretation of confidence level
    """
    if confidence >= 0.9:
        return "Very High - Strong consensus across all LLMs"
    elif confidence >= 0.75:
        return "High - Majority agreement with few disputes"
    elif confidence >= 0.6:
        return "Moderate - Reasonable consensus with some disagreement"
    elif confidence >= 0.4:
        return "Low - Significant disagreement among LLMs"
    else:
        return "Very Low - Little to no consensus, high uncertainty"
