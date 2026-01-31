#!/usr/bin/env python3
"""
Example Usage Script

Demonstrates various ways to use the Multi-LLM Arbitration Engine.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arbitration_engine.pipeline import run_pipeline, ArbitrationPipeline
from arbitration_engine.agents import run_interpreter_agent
from arbitration_engine.arbitration import (
    extract_claims,
    compute_claim_similarity,
    interpret_confidence
)


def example_1_basic_usage():
    """Example 1: Basic pipeline usage"""
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)
    
    question = "What is machine learning?"
    
    print(f"\nQuestion: {question}\n")
    
    try:
        result = run_pipeline(question)
        
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Interpretation: {result['confidence_interpretation']}")
        print(f"\nFinal Answer:\n{result['final_answer']}")
        print(f"\nStatistics:")
        print(f"  - Total claims: {result['statistics']['total_unique_claims']}")
        print(f"  - Supported: {result['statistics']['num_supported']}")
        print(f"  - Disputed: {result['statistics']['num_disputed']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def example_2_detailed_analysis():
    """Example 2: Detailed claim analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Detailed Claim Analysis")
    print("="*80)
    
    question = "How does blockchain technology work?"
    
    print(f"\nQuestion: {question}\n")
    
    try:
        result = run_pipeline(question)
        
        print("Claim Analysis by LLM:")
        for llm_name, claims in result['claim_sets'].items():
            print(f"\n{llm_name.upper()} ({len(claims)} claims):")
            for i, claim in enumerate(claims[:3], 1):
                print(f"  {i}. {claim[:80]}...")
        
        print("\nPairwise Agreement Scores:")
        for pair, score in result['pairwise_agreement'].items():
            print(f"  {pair}: {score:.3f}")
        
        print("\nUnanimous Claims (all LLMs agree):")
        for claim in result['unanimous_claims'][:5]:
            print(f"  ✓ {claim}")
        
        print("\nDisputed Claims (single LLM only):")
        for claim in result['disputed_claims'][:5]:
            print(f"  ✗ {claim}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def example_3_interpreter_only():
    """Example 3: Using interpreter agent standalone"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Interpreter Agent Only")
    print("="*80)
    
    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity",
        "Compare democracy and authoritarianism",
    ]
    
    print("\nAnalyzing question metadata:\n")
    
    for question in questions:
        metadata = run_interpreter_agent(question)
        print(f"Q: {question}")
        print(f"  Concept: {metadata['concept']}")
        print(f"  Intent: {metadata['intent']}")
        print(f"  Difficulty: {metadata['difficulty']}")
        print(f"  Keywords: {', '.join(metadata['keywords'][:5])}")
        print()


def example_4_claim_similarity():
    """Example 4: Claim similarity analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Claim Similarity Analysis")
    print("="*80)
    
    claims = [
        "The Earth orbits around the Sun",
        "Our planet revolves around the Sun",
        "The Moon orbits around Earth",
        "Earth is the third planet from the Sun",
    ]
    
    print("\nComparing claims:\n")
    
    for i, claim1 in enumerate(claims):
        for claim2 in claims[i+1:]:
            similarity = compute_claim_similarity(claim1, claim2)
            print(f"Similarity: {similarity:.3f}")
            print(f"  1: {claim1}")
            print(f"  2: {claim2}")
            print()


def example_5_custom_prompt():
    """Example 5: Using custom prompt template"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Prompt Template")
    print("="*80)
    
    # Create custom prompt
    custom_prompt = """You are a scientific expert. Answer precisely and factually.

Question: {question}

Provide a detailed scientific answer with specific facts:"""
    
    # Save to temporary file
    custom_prompt_path = Path("custom_prompt_temp.txt")
    custom_prompt_path.write_text(custom_prompt)
    
    try:
        # Initialize pipeline with custom prompt
        pipeline = ArbitrationPipeline(canonical_prompt_path=str(custom_prompt_path))
        
        question = "What is DNA?"
        print(f"\nQuestion: {question}\n")
        
        result = pipeline.run(question)
        
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"\nFinal Answer:\n{result['final_answer']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean up
        if custom_prompt_path.exists():
            custom_prompt_path.unlink()


def example_6_save_results():
    """Example 6: Saving results to file"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Saving Results")
    print("="*80)
    
    question = "What causes earthquakes?"
    
    print(f"\nQuestion: {question}\n")
    
    try:
        result = run_pipeline(question)
        
        # Save to JSON
        output_file = "example_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        print(f"File size: {Path(output_file).stat().st_size} bytes")
        
        # Display summary
        print(f"\nSummary:")
        print(f"  Confidence: {result['confidence_score']:.3f}")
        print(f"  Total claims: {result['statistics']['total_unique_claims']}")
        print(f"  LLMs used: {', '.join(result['statistics']['llms_used'])}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def main():
    """Run all examples"""
    
    print("\n" + "="*80)
    print("Multi-LLM Arbitration Engine - Example Usage")
    print("="*80)
    
    # Check environment
    if not os.environ.get('OPENAI_API_KEY'):
        print("\nWARNING: OPENAI_API_KEY not set. Some examples may fail.")
        print("Set with: export OPENAI_API_KEY='your-key'\n")
    
    if not os.environ.get('COHERE_API_KEY'):
        print("\nWARNING: COHERE_API_KEY not set. Some examples may fail.")
        print("Set with: export COHERE_API_KEY='your-key'\n")
    
    # Run examples
    try:
        # example_1_basic_usage()
        # example_2_detailed_analysis()
        example_3_interpreter_only()
        example_4_claim_similarity()
        # example_5_custom_prompt()
        # example_6_save_results()
        
        print("\n" + "="*80)
        print("All examples completed!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nExample failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
