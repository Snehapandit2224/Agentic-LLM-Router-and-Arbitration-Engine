"""
Arbitration Module

Contains components for claim extraction, agreement analysis, and confidence scoring.
"""

from .claim_extractor import (
    extract_claims,
    ClaimExtractor,
    compute_claim_similarity,
    find_similar_claims
)

from .agreement_engine import (
    build_agreement_table,
    categorize_claims,
    AgreementEngine,
    get_claim_supporters,
    calculate_pairwise_agreement
)

from .confidence_scorer import (
    compute_confidence,
    compute_advanced_confidence,
    ConfidenceScorer,
    interpret_confidence
)

__all__ = [
    'extract_claims',
    'ClaimExtractor',
    'compute_claim_similarity',
    'find_similar_claims',
    'build_agreement_table',
    'categorize_claims',
    'AgreementEngine',
    'get_claim_supporters',
    'calculate_pairwise_agreement',
    'compute_confidence',
    'compute_advanced_confidence',
    'ConfidenceScorer',
    'interpret_confidence'
]
