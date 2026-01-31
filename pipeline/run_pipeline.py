"""
Pipeline Orchestration Module

Coordinates the complete multi-LLM arbitration pipeline.
"""

import os
import sys
import logging
from typing import Dict, Set, List, Tuple, Optional
from pathlib import Path
import json

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    run_interpreter_agent,
    call_llm_a,
    call_llm_b,
    call_llm_c,
    APIKeyMissingError,
    LLMAPIError
)

from arbitration import (
    extract_claims,
    build_agreement_table,
    categorize_claims,
    compute_confidence,
    compute_advanced_confidence,
    interpret_confidence,
    # calculate_pairwise_agreement
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cluster_based_pairwise_agreement(clusters, claim_sets, llm_a, llm_b):
        """
        Compute pairwise agreement using semantic clusters.
        """
        shared = 0
        total = len(clusters)

        for cluster in clusters:
            has_a = any(c in claim_sets[llm_a] for c in cluster)
            has_b = any(c in claim_sets[llm_b] for c in cluster)

            if has_a and has_b:
                shared += 1

        return shared / total if total > 0 else 0.0

class ArbitrationPipeline:
    """Orchestrates the multi-LLM arbitration pipeline."""
    
    def __init__(self, canonical_prompt_path: Optional[str] = None):
        """
        Initialize the arbitration pipeline.
        
        Args:
            canonical_prompt_path: Path to canonical prompt template file
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load canonical prompt
        if canonical_prompt_path is None:
            # Default to prompts directory relative to this file
            canonical_prompt_path = Path(__file__).parent.parent / "prompts" / "canonical_prompt.txt"
        
        self.canonical_prompt_template = self._load_prompt_template(canonical_prompt_path)
        
        # LLM configuration
        self.llm_clients = {
            'cohere': call_llm_b,
            'GROQ': call_llm_c
        }



    
    def _load_prompt_template(self, path: Path) -> str:
        """Load the canonical prompt template from file."""
        try:
            with open(path, 'r') as f:
                template = f.read()
            self.logger.info(f"Loaded canonical prompt template from {path}")
            return template
        except FileNotFoundError:
            self.logger.error(f"Canonical prompt template not found at {path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {str(e)}")
            raise
    
    def _build_canonical_prompt(self, question: str) -> str:
        """Build the canonical prompt by inserting the question."""
        return self.canonical_prompt_template.format(question=question)
    
    def _call_all_llms(self, prompt: str) -> Dict[str, str]:
        """
        Call all three LLMs with the same prompt.
        
        Args:
            prompt: The canonical prompt to send
            
        Returns:
            Dictionary mapping LLM names to their responses
        """
        responses = {}
        errors = {}
        
        for llm_name, llm_client in self.llm_clients.items():
            try:
                self.logger.info(f"Calling {llm_name}...")
                response = llm_client(prompt)
                responses[llm_name] = response
                self.logger.info(f"{llm_name} response received: {len(response)} chars")
            
            except (APIKeyMissingError, LLMAPIError) as e:
                self.logger.error(f"{llm_name} failed: {str(e)}")
                errors[llm_name] = str(e)
        
        if not responses:
            error_msg = "All LLM calls failed:\n" + "\n".join(
                f"  - {llm}: {err}" for llm, err in errors.items()
            )
            raise RuntimeError(error_msg)
        
        if len(responses) < len(self.llm_clients):
            self.logger.warning(
                f"Only {len(responses)}/{len(self.llm_clients)} LLMs responded successfully"
            )
        
        return responses
    
    def _extract_all_claims(self, responses: Dict[str, str]) -> Dict[str, Set[str]]:
        """
        Extract claims from all LLM responses.
        
        Args:
            responses: Dictionary of LLM responses
            
        Returns:
            Dictionary mapping LLM names to their claim sets
        """
        claim_sets = {}
        
        for llm_name, response in responses.items():
            try:
                claims = extract_claims(response)
                claim_sets[llm_name] = claims
                self.logger.info(f"Extracted {len(claims)} claims from {llm_name}")
            except Exception as e:
                self.logger.error(f"Failed to extract claims from {llm_name}: {str(e)}")
                claim_sets[llm_name] = set()
        
        return claim_sets
    
    def _merge_supported_claims(self, supported_claims: Set[str]) -> str:
        """
        Merge supported claims into a coherent final answer.
        
        Args:
            supported_claims: Set of claims with consensus support
            
        Returns:
            Merged answer string
        """
        if not supported_claims:
            return "Unable to generate consensus answer - no supported claims found."
        
        # Sort claims for consistent output
        sorted_claims = sorted(supported_claims)
        
        # Simple merge: join with proper punctuation
        merged = ". ".join(claim.capitalize() for claim in sorted_claims)
        
        # Ensure final period
        if not merged.endswith('.'):
            merged += '.'
        
        return merged
    
    def run(self, question: str) -> Dict[str, any]:
        """
        Run the complete arbitration pipeline.
        
        Args:
            question: User's input question
            
        Returns:
            Dictionary containing:
                - question: Original question
                - interpreter_metadata: Metadata from interpreter agent
                - llm_responses: Raw responses from each LLM
                - claim_sets: Extracted claims from each LLM
                - agreement_table: Mapping of claims to supporting LLMs
                - supported_claims: Claims with consensus
                - disputed_claims: Claims from only one LLM
                - unanimous_claims: Claims from all LLMs
                - confidence_score: Overall confidence score
                - confidence_interpretation: Human-readable confidence level
                - final_answer: Merged consensus answer
                - pairwise_agreement: Agreement scores between LLM pairs
        """
        self.logger.info("="*80)
        self.logger.info(f"Starting arbitration pipeline for question: {question}")
        self.logger.info("="*80)
        
        try:
            # Step 1: Interpreter Agent
            self.logger.info("Step 1: Running interpreter agent...")
            interpreter_metadata = run_interpreter_agent(question)
            
            # Step 2: Build canonical prompt
            self.logger.info("Step 2: Building canonical prompt...")
            canonical_prompt = self._build_canonical_prompt(question)
            
            # Step 3: Call all LLMs
            self.logger.info("Step 3: Calling all LLMs...")
            llm_responses = self._call_all_llms(canonical_prompt)
            
            # Step 4: Extract claims
            self.logger.info("Step 4: Extracting claims...")
            claim_sets = self._extract_all_claims(llm_responses)
            
            # Step 5: Build agreement table
            self.logger.info("Step 5: Building agreement table...")
            agreement_table = build_agreement_table(llm_responses, claim_sets)
            
            # Step 6: Categorize claims
            self.logger.info("Step 6: Categorizing claims...")
            supported_claims, disputed_claims, unanimous_claims = categorize_claims(agreement_table)
            
            # Step 7: Compute confidence
            self.logger.info("Step 7: Computing confidence scores...")
            total_claims = len(agreement_table)
            basic_confidence = compute_confidence(supported_claims, total_claims)
            advanced_confidence = compute_advanced_confidence(
                supported_claims, disputed_claims, unanimous_claims, total_claims
            )
            
            # Use advanced confidence as primary score
            confidence_score = advanced_confidence
            confidence_interpretation = interpret_confidence(confidence_score)
            
            # Step 8: Calculate pairwise agreement (SEMANTIC)
            self.logger.info("Step 8: Computing pairwise agreement...")

            # Build semantic clusters ONCE (same basis as agreement table)
            all_claims = list(set().union(*claim_sets.values()))

            from arbitration.semantic_cluster import cluster_claims
            from arbitration.agreement_engine import embed_claims

            clusters = cluster_claims(
                all_claims,
                embed_fn=embed_claims,
                threshold=0.6            )

            pairwise_agreement = {}

            llms = list(claim_sets.keys())
            for i in range(len(llms)):
                for j in range(i + 1, len(llms)):
                    llm_a, llm_b = llms[i], llms[j]
                    pairwise_agreement[(llm_a, llm_b)] = cluster_based_pairwise_agreement(
                        clusters,
                        claim_sets,
                        llm_a,
                        llm_b
                    )

            
            # Step 9: Merge final answer
            self.logger.info("Step 9: Merging final answer...")
            final_answer = self._merge_supported_claims(supported_claims)
            
            # Build result
            result = {
                'question': question,
                'interpreter_metadata': interpreter_metadata,
                'llm_responses': llm_responses,
                'claim_sets': {k: list(v) for k, v in claim_sets.items()},  # Convert sets to lists for JSON
                'agreement_table': agreement_table,
                'supported_claims': list(supported_claims),
                'disputed_claims': list(disputed_claims),
                'unanimous_claims': list(unanimous_claims),
                'confidence_score': round(confidence_score, 3),
                'basic_confidence': round(basic_confidence, 3),
                'confidence_interpretation': confidence_interpretation,
                'final_answer': final_answer,
                'pairwise_agreement': {
                    f"{llm1}_vs_{llm2}": round(score, 3)
                    for (llm1, llm2), score in pairwise_agreement.items()
                },
                'statistics': {
                    'total_unique_claims': total_claims,
                    'num_supported': len(supported_claims),
                    'num_disputed': len(disputed_claims),
                    'num_unanimous': len(unanimous_claims),
                    'llms_used': list(llm_responses.keys())
                }
            }
            
            self.logger.info("="*80)
            self.logger.info("Pipeline completed successfully")
            self.logger.info(f"Confidence: {confidence_score:.3f} ({confidence_interpretation})")
            self.logger.info(f"Supported claims: {len(supported_claims)}/{total_claims}")
            self.logger.info("="*80)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def run_pipeline(question: str) -> Dict[str, any]:
    """
    Main entry point for running the arbitration pipeline.
    
    Args:
        question: User's input question
        
    Returns:
        Complete arbitration result dictionary
    """
    pipeline = ArbitrationPipeline()
    return pipeline.run(question)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Demonstration of the arbitration pipeline."""
    
    print("="*80)
    print("Multi-LLM Arbitration Engine - Demo")
    print("="*80)
    print()
    
    # Check environment variables
    print("Checking environment configuration...")
    required_vars = ['OPENAI_API_KEY', 'COHERE_API_KEY', 'GROQ_API_KEY']
    optional_vars = []
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
            print(f"  ✗ {var} - NOT SET")
        else:
            print(f"  ✓ {var} - configured")
    
    for var in optional_vars:
        value = os.environ.get(var, 'http://localhost:11434')
        print(f"  ✓ {var} - {value}")
    
    if missing_vars:
        print()
        print("ERROR: Missing required environment variables:")
        for var in missing_vars:
            print(f"  export {var}='your-key-here'")
        print()
        sys.exit(1)
    
    print()
    print("Environment configured successfully!")
    print()
    
    try:
        while True:
            print("="*80)
            print("Enter your question (or 'exit' to quit):")
            print("="*80)
            question = input("\n>> ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using the Arbitration Engine!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            print()
            print("="*80)
            print(f"Processing: {question}")
            print("="*80)
            print()
            
            # Run pipeline
            result = run_pipeline(question)
            
            # Display results
            print()
            print("="*80)
            print("ARBITRATION RESULTS")
            print("="*80)
            print()
            
            print(f"Question: {result['question']}")
            print()
            
            print("LLM Responses:")
            for llm_name, response in result['llm_responses'].items():
                print(f"\n{llm_name.upper()}:")
                print(f"  {response[:200]}..." if len(response) > 200 else f"  {response}")
            print()
            
            print("Claim Analysis:")
            print(f"  Total unique claims: {result['statistics']['total_unique_claims']}")
            print(f"  Unanimous claims: {result['statistics']['num_unanimous']}")
            print(f"  Supported claims: {result['statistics']['num_supported']}")
            print(f"  Disputed claims: {result['statistics']['num_disputed']}")
            print()
            
            print("Pairwise Agreement:")
            for pair, score in result['pairwise_agreement'].items():
                print(f"  {pair}: {score:.3f}")
            print()
            
            print(f"Confidence Score: {result['confidence_score']:.3f}")
            print(f"Interpretation: {result['confidence_interpretation']}")
            print()
            
            print("FINAL CONSENSUS ANSWER:")
            print(f"  {result['final_answer']}")
            print()
            
            print("Disputed Claims (no consensus):")
            if result['disputed_claims']:
                for claim in result['disputed_claims'][:5]:  # Show first 5
                    print(f"  - {claim}")
                if len(result['disputed_claims']) > 5:
                    print(f"  ... and {len(result['disputed_claims']) - 5} more")
            else:
                print("  None")
            print()
            
            # Option to save results
            save = input("Save detailed results to JSON? (y/n): ").strip().lower()
            if save == 'y':
                output_file = f"arbitration_result_{hash(question) % 10000}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Results saved to: {output_file}")
            print()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        print(f"\nError processing question: {str(e)}")
        print("Please check your API keys and try again.")
        print()


if __name__ == "__main__":
    main()
