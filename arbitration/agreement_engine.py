"""
Agreement Engine Module

Analyzes claim agreement and disagreement across multiple LLM responses.
"""
from arbitration.semantic_cluster import cluster_claims
from sentence_transformers import SentenceTransformer
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_claims(claims):
    return _embedding_model.encode(claims)


import logging
from typing import Dict, Set, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class AgreementEngine:
    """Analyzes consensus and disagreement across LLM responses."""
    
    def __init__(self, consensus_threshold: int = 2):
        """
        Initialize the agreement engine.
        
        Args:
            consensus_threshold: Minimum number of LLMs that must agree on a claim
        """
        self.consensus_threshold = consensus_threshold
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def build_agreement_table(
        self, 
        responses: Dict[str, str], 
        claim_sets: Dict[str, Set[str]]
    ) -> Dict[str, List[str]]:
        """
        Build a table mapping claims to the LLMs that support them.
        
        Args:
            responses: Dictionary mapping LLM names to their responses
            claim_sets: Dictionary mapping LLM names to their extracted claims
            
        Returns:
            Dictionary mapping claims to list of supporting LLM names
        """
        if not responses or not claim_sets:
            raise ValueError("Responses and claim_sets cannot be empty")
        
        if set(responses.keys()) != set(claim_sets.keys()):
            raise ValueError("Response keys must match claim_sets keys")
        
        self.logger.info("Building agreement table")
        
        # agreement_table = defaultdict(list)
        
        # # For each LLM's claims, record which LLM supports each claim
        # for llm_name, claims in claim_sets.items():
        #     for claim in claims:
        #         agreement_table[claim].append(llm_name)

        agreement_table = defaultdict(list)

        # 1. Collect all unique claims
        all_claims = list(set().union(*claim_sets.values()))

        # 2. Cluster semantically similar claims
        clusters = cluster_claims(
            all_claims,
            embed_fn=embed_claims,
            threshold=0.8
        )

        # 3. Build agreement table using clusters
        for cluster in clusters:
            canonical_claim = cluster[0]  # representative claim

            for llm_name, claims in claim_sets.items():
                if any(c in claims for c in cluster):
                    agreement_table[canonical_claim].append(llm_name)

        
        # Convert to regular dict
        agreement_table = dict(agreement_table)
        
        self.logger.info(f"Agreement table built with {len(agreement_table)} unique claims")
        return agreement_table
    
    def categorize_claims(
        self, 
        agreement_table: Dict[str, List[str]]
    ) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Categorize claims into supported, disputed, and unanimous.
        
        Args:
            agreement_table: Mapping of claims to supporting LLMs
            
        Returns:
            Tuple of (supported_claims, disputed_claims, unanimous_claims)
            - supported_claims: Claims supported by >= consensus_threshold LLMs
            - disputed_claims: Claims supported by exactly 1 LLM
            - unanimous_claims: Claims supported by all LLMs
        """
        if not agreement_table:
            self.logger.warning("Empty agreement table provided")
            return set(), set(), set()
        
        self.logger.info("Categorizing claims by agreement level")
        
        supported_claims = set()
        disputed_claims = set()
        unanimous_claims = set()
        
        # Determine total number of LLMs
        max_supporters = max(len(supporters) for supporters in agreement_table.values())
        
        for claim, supporters in agreement_table.items():
            num_supporters = len(supporters)
            
            # Unanimous: all LLMs agree
            if num_supporters == max_supporters:
                unanimous_claims.add(claim)
                supported_claims.add(claim)
            
            # Supported: meets consensus threshold
            elif num_supporters >= self.consensus_threshold:
                supported_claims.add(claim)
            
            # Disputed: only 1 LLM supports
            elif num_supporters == 1:
                disputed_claims.add(claim)
        
        self.logger.info(
            f"Categorized claims - "
            f"Unanimous: {len(unanimous_claims)}, "
            f"Supported: {len(supported_claims)}, "
            f"Disputed: {len(disputed_claims)}"
        )
        
        return supported_claims, disputed_claims, unanimous_claims
    
    def generate_agreement_report(
        self,
        agreement_table: Dict[str, List[str]],
        supported_claims: Set[str],
        disputed_claims: Set[str],
        unanimous_claims: Set[str]
    ) -> Dict[str, any]:
        """
        Generate a detailed agreement analysis report.
        
        Args:
            agreement_table: Claim to supporters mapping
            supported_claims: Set of supported claims
            disputed_claims: Set of disputed claims
            unanimous_claims: Set of unanimous claims
            
        Returns:
            Dictionary containing detailed agreement statistics
        """
        total_claims = len(agreement_table)
        
        # Calculate agreement distribution
        agreement_distribution = defaultdict(int)
        for supporters in agreement_table.values():
            agreement_distribution[len(supporters)] += 1
        
        # Identify highly disputed claims (with supporting LLMs)
        disputed_details = {}
        for claim in disputed_claims:
            supporters = agreement_table.get(claim, [])
            if supporters:
                disputed_details[claim] = supporters[0]
        
        report = {
            'total_unique_claims': total_claims,
            'unanimous_claims': len(unanimous_claims),
            'supported_claims': len(supported_claims),
            'disputed_claims': len(disputed_claims),
            'agreement_distribution': dict(agreement_distribution),
            'consensus_threshold': self.consensus_threshold,
            'unanimous_percentage': (
                len(unanimous_claims) / total_claims * 100 
                if total_claims > 0 else 0
            ),
            'disputed_details': disputed_details
        }
        
        self.logger.debug(f"Generated agreement report: {report}")
        return report


def build_agreement_table(
    responses: Dict[str, str], 
    claim_sets: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """
    Main entry point for building agreement table.
    
    Args:
        responses: Dictionary mapping LLM names to their text responses
        claim_sets: Dictionary mapping LLM names to their claim sets
        
    Returns:
        Dictionary mapping claims to list of supporting LLM names
    """
    engine = AgreementEngine()
    return engine.build_agreement_table(responses, claim_sets)


def categorize_claims(agreement_table: Dict[str, List[str]]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Main entry point for categorizing claims.
    
    Args:
        agreement_table: Mapping of claims to supporting LLMs
        
    Returns:
        Tuple of (supported_claims, disputed_claims, unanimous_claims)
    """
    engine = AgreementEngine()
    return engine.categorize_claims(agreement_table)


# ============================================================================
# Utility Functions
# ============================================================================

def get_claim_supporters(claim: str, agreement_table: Dict[str, List[str]]) -> List[str]:
    """
    Get the list of LLMs that support a specific claim.
    
    Args:
        claim: The claim to look up
        agreement_table: Agreement table mapping
        
    Returns:
        List of LLM names supporting the claim
    """
    return agreement_table.get(claim, [])


def calculate_pairwise_agreement(
    claim_sets: Dict[str, Set[str]]
) -> Dict[Tuple[str, str], float]:
    """
    Calculate pairwise agreement scores between all LLM pairs.
    
    Args:
        claim_sets: Dictionary mapping LLM names to their claim sets
        
    Returns:
        Dictionary mapping (llm1, llm2) tuples to Jaccard similarity scores
    """
    llm_names = list(claim_sets.keys())
    pairwise_scores = {}
    
    for i, llm1 in enumerate(llm_names):
        for llm2 in llm_names[i+1:]:
            claims1 = claim_sets[llm1]
            claims2 = claim_sets[llm2]
            
            if not claims1 or not claims2:
                similarity = 0.0
            else:
                intersection = len(claims1 & claims2)
                union = len(claims1 | claims2)
                similarity = intersection / union if union > 0 else 0.0
            
            pairwise_scores[(llm1, llm2)] = similarity
    
    return pairwise_scores
