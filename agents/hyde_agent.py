"""
Hypothetical Document Embeddings (HyDE) Agent

Generates expert-style explanatory content based on question analysis.
"""

import os
import logging
from typing import Dict
from .llm_clients import call_llm_a

logger = logging.getLogger(__name__)


class HydeAgent:
    """Generates hypothetical expert responses using HyDE methodology."""
    
    HYDE_PROMPT_TEMPLATE = """You are an expert providing a concise, authoritative explanation.

Question: {question}

Context:
- Domain: {concept}
- Intent: {intent}
- Difficulty: {difficulty}
- Key terms: {keywords}

Provide a brief, factual expert explanation (2-3 paragraphs maximum).
Focus on accuracy and clarity. Do not speculate or add caveats."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate(self, interpreter_output: Dict[str, any], question: str) -> str:
        """
        Generate a hypothetical expert document.
        
        Args:
            interpreter_output: Metadata from interpreter agent
            question: Original user question
            
        Returns:
            Generated expert-style explanation text
        """
        if not interpreter_output or not question:
            raise ValueError("Both interpreter_output and question are required")
        
        self.logger.info("Generating HyDE document")
        
        try:
            # Build the HyDE prompt
            keywords_str = ', '.join(interpreter_output.get('keywords', [])[:5])
            
            prompt = self.HYDE_PROMPT_TEMPLATE.format(
                question=question,
                concept=interpreter_output.get('concept', 'general'),
                intent=interpreter_output.get('intent', 'query'),
                difficulty=interpreter_output.get('difficulty', 'medium'),
                keywords=keywords_str or 'N/A'
            )
            
            self.logger.debug(f"HyDE prompt length: {len(prompt)} chars")
            
            # Generate using LLM A (OpenAI)
            response = call_llm_a(prompt)
            
            if not response:
                raise RuntimeError("HyDE agent received empty response from LLM")
            
            self.logger.info(f"Generated HyDE document: {len(response)} chars")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating HyDE document: {str(e)}")
            raise


def run_hyde_agent(interpreter_output: Dict[str, any], question: str) -> str:
    """
    Main entry point for HyDE agent.
    
    Args:
        interpreter_output: Metadata dictionary from interpreter agent
        question: Original user question
        
    Returns:
        Generated expert explanation string
    """
    agent = HydeAgent()
    return agent.generate(interpreter_output, question)
