"""
Interpreter Agent Module

Analyzes user questions to extract metadata for downstream processing.
"""

import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class InterpreterAgent:
    """Analyzes and interprets user questions to extract structured metadata."""
    
    DIFFICULTY_KEYWORDS = {
        'easy': ['what is', 'define', 'who is', 'when did', 'where is'],
        'medium': ['how does', 'explain', 'describe', 'compare', 'why'],
        'hard': ['analyze', 'evaluate', 'synthesize', 'critique', 'prove']
    }
    
    CONCEPT_PATTERNS = {
        'scientific': r'\b(atom|molecule|cell|gene|DNA|physics|chemistry|biology|quantum)\b',
        'mathematical': r'\b(equation|theorem|proof|calculate|derivative|integral|algebra)\b',
        'historical': r'\b(war|empire|revolution|century|ancient|medieval|treaty)\b',
        'technical': r'\b(algorithm|code|program|network|system|database|API)\b',
        'philosophical': r'\b(ethics|morality|consciousness|existence|knowledge|truth)\b',
        'economic': r'\b(market|economy|trade|GDP|inflation|supply|demand)\b',
    }
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _detect_intent(self, question: str) -> str:
        """Detect the primary intent of the question."""
        question_lower = question.lower().strip()
        
        if question_lower.startswith(('what', 'define', 'who is')):
            return 'definition'
        elif question_lower.startswith(('how', 'explain', 'describe')):
            return 'explanation'
        elif question_lower.startswith(('why', 'what causes')):
            return 'causation'
        elif 'compare' in question_lower or 'difference' in question_lower:
            return 'comparison'
        elif any(word in question_lower for word in ['should', 'recommend', 'best']):
            return 'recommendation'
        else:
            return 'general_query'
    
    def _assess_difficulty(self, question: str) -> str:
        """Assess the difficulty level of the question."""
        question_lower = question.lower()
        
        for difficulty, keywords in self.DIFFICULTY_KEYWORDS.items():
            if any(keyword in question_lower for keyword in keywords):
                return difficulty
        
        # Default to medium if no clear indicators
        return 'medium'
    
    def _extract_concept(self, question: str) -> str:
        """Extract the primary conceptual domain of the question."""
        for concept, pattern in self.CONCEPT_PATTERNS.items():
            if re.search(pattern, question, re.IGNORECASE):
                return concept
        
        return 'general'
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract key terms from the question."""
        # Remove common question words
        stop_words = {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 
                      'where', 'who', 'which', 'does', 'do', 'did', 'can', 
                      'could', 'should', 'would', 'are', 'was', 'were', 'be'}
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', question.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Return unique keywords, maintaining order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # Limit to top 10
    
    def analyze(self, question: str) -> Dict[str, any]:
        """
        Analyze the question and return structured metadata.
        
        Args:
            question: The user's input question
            
        Returns:
            Dictionary containing:
                - concept: Primary conceptual domain
                - intent: Detected question intent
                - difficulty: Assessed difficulty level
                - keywords: List of extracted keywords
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        self.logger.info(f"Analyzing question: {question[:100]}...")
        
        try:
            metadata = {
                'concept': self._extract_concept(question),
                'intent': self._detect_intent(question),
                'difficulty': self._assess_difficulty(question),
                'keywords': self._extract_keywords(question)
            }
            
            self.logger.debug(f"Extracted metadata: {metadata}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error analyzing question: {str(e)}")
            raise


def run_interpreter_agent(question: str) -> Dict[str, any]:
    """
    Main entry point for the interpreter agent.
    
    Args:
        question: The user's question
        
    Returns:
        Structured metadata dictionary
    """
    agent = InterpreterAgent()
    return agent.analyze(question)
