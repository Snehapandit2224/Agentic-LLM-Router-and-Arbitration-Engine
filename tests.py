#!/usr/bin/env python3
"""
Test Suite for Multi-LLM Arbitration Engine

Run with: python -m pytest tests.py -v
or: python tests.py
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from arbitration_engine.agents import (
    InterpreterAgent,
    run_interpreter_agent,
    APIKeyMissingError,
    LLMAPIError
)

from arbitration_engine.arbitration import (
    ClaimExtractor,
    extract_claims,
    build_agreement_table,
    categorize_claims,
    compute_confidence,
    compute_claim_similarity
)


class TestInterpreterAgent(unittest.TestCase):
    """Test cases for InterpreterAgent"""
    
    def setUp(self):
        self.agent = InterpreterAgent()
    
    def test_detect_intent_definition(self):
        question = "What is machine learning?"
        metadata = self.agent.analyze(question)
        self.assertEqual(metadata['intent'], 'definition')
    
    def test_detect_intent_explanation(self):
        question = "How does photosynthesis work?"
        metadata = self.agent.analyze(question)
        self.assertEqual(metadata['intent'], 'explanation')
    
    def test_detect_intent_comparison(self):
        question = "Compare supervised and unsupervised learning"
        metadata = self.agent.analyze(question)
        self.assertEqual(metadata['intent'], 'comparison')
    
    def test_assess_difficulty_easy(self):
        question = "What is water?"
        metadata = self.agent.analyze(question)
        self.assertEqual(metadata['difficulty'], 'easy')
    
    def test_assess_difficulty_hard(self):
        question = "Analyze the ethical implications of AGI"
        metadata = self.agent.analyze(question)
        self.assertEqual(metadata['difficulty'], 'hard')
    
    def test_extract_concept_scientific(self):
        question = "How do molecules interact?"
        metadata = self.agent.analyze(question)
        self.assertEqual(metadata['concept'], 'scientific')
    
    def test_extract_keywords(self):
        question = "What is quantum computing and how does it work?"
        metadata = self.agent.analyze(question)
        self.assertIn('quantum', metadata['keywords'])
        self.assertIn('computing', metadata['keywords'])
    
    def test_empty_question_raises_error(self):
        with self.assertRaises(ValueError):
            self.agent.analyze("")
    
    def test_non_string_raises_error(self):
        with self.assertRaises(ValueError):
            self.agent.analyze(123)


class TestClaimExtractor(unittest.TestCase):
    """Test cases for ClaimExtractor"""
    
    def setUp(self):
        self.extractor = ClaimExtractor()
    
    def test_extract_single_claim(self):
        text = "Water is composed of hydrogen and oxygen."
        claims = self.extractor.extract(text)
        self.assertEqual(len(claims), 1)
        self.assertIn("water is composed of hydrogen and oxygen", claims)
    
    def test_extract_multiple_claims(self):
        text = "The Earth is round. It orbits the Sun. The Moon orbits Earth."
        claims = self.extractor.extract(text)
        self.assertGreaterEqual(len(claims), 2)
    
    def test_normalize_claim(self):
        claim = "  The SKY is BLUE.  "
        normalized = self.extractor._normalize_claim(claim)
        self.assertEqual(normalized, "the sky is blue")
    
    def test_filter_short_claims(self):
        text = "Hi. This is a proper sentence with sufficient length."
        claims = self.extractor.extract(text)
        # "Hi" should be filtered out
        self.assertNotIn("hi", claims)
    
    def test_empty_text_returns_empty_set(self):
        claims = self.extractor.extract("")
        self.assertEqual(len(claims), 0)
    
    def test_sentence_splitting(self):
        text = "Dr. Smith said the experiment works. It was successful."
        sentences = self.extractor._split_into_sentences(text)
        self.assertGreaterEqual(len(sentences), 2)


class TestClaimSimilarity(unittest.TestCase):
    """Test cases for claim similarity computation"""
    
    def test_identical_claims(self):
        claim1 = "the earth is round"
        claim2 = "the earth is round"
        similarity = compute_claim_similarity(claim1, claim2)
        self.assertEqual(similarity, 1.0)
    
    def test_similar_claims(self):
        claim1 = "the earth orbits the sun"
        claim2 = "earth revolves around the sun"
        similarity = compute_claim_similarity(claim1, claim2)
        self.assertGreater(similarity, 0.4)
    
    def test_dissimilar_claims(self):
        claim1 = "the sky is blue"
        claim2 = "grass is green"
        similarity = compute_claim_similarity(claim1, claim2)
        self.assertLess(similarity, 0.3)
    
    def test_empty_claims(self):
        similarity = compute_claim_similarity("", "test")
        self.assertEqual(similarity, 0.0)


class TestAgreementEngine(unittest.TestCase):
    """Test cases for agreement analysis"""
    
    def test_build_agreement_table(self):
        responses = {
            'llm1': 'The sky is blue. Water is wet.',
            'llm2': 'The sky is blue. Fire is hot.',
            'llm3': 'The sky is blue. Ice is cold.'
        }
        
        claim_sets = {
            'llm1': {'the sky is blue', 'water is wet'},
            'llm2': {'the sky is blue', 'fire is hot'},
            'llm3': {'the sky is blue', 'ice is cold'}
        }
        
        table = build_agreement_table(responses, claim_sets)
        
        # "the sky is blue" should be supported by all 3
        self.assertEqual(len(table['the sky is blue']), 3)
        
        # Other claims should be supported by 1 each
        self.assertEqual(len(table['water is wet']), 1)
    
    def test_categorize_claims_unanimous(self):
        agreement_table = {
            'claim1': ['llm1', 'llm2', 'llm3'],
            'claim2': ['llm1', 'llm2'],
            'claim3': ['llm1']
        }
        
        supported, disputed, unanimous = categorize_claims(agreement_table)
        
        self.assertEqual(len(unanimous), 1)
        self.assertIn('claim1', unanimous)
        self.assertIn('claim1', supported)
        self.assertIn('claim2', supported)
        self.assertIn('claim3', disputed)
    
    def test_empty_agreement_table(self):
        supported, disputed, unanimous = categorize_claims({})
        self.assertEqual(len(supported), 0)
        self.assertEqual(len(disputed), 0)
        self.assertEqual(len(unanimous), 0)


class TestConfidenceScorer(unittest.TestCase):
    """Test cases for confidence scoring"""
    
    def test_perfect_confidence(self):
        supported_claims = {'claim1', 'claim2', 'claim3'}
        total_claims = 3
        confidence = compute_confidence(supported_claims, total_claims)
        self.assertEqual(confidence, 1.0)
    
    def test_zero_confidence(self):
        supported_claims = set()
        total_claims = 5
        confidence = compute_confidence(supported_claims, total_claims)
        self.assertEqual(confidence, 0.0)
    
    def test_partial_confidence(self):
        supported_claims = {'claim1', 'claim2'}
        total_claims = 4
        confidence = compute_confidence(supported_claims, total_claims)
        self.assertEqual(confidence, 0.5)
    
    def test_zero_total_claims(self):
        confidence = compute_confidence(set(), 0)
        self.assertEqual(confidence, 0.0)


class TestLLMClients(unittest.TestCase):
    """Test cases for LLM client error handling"""
    
    @patch.dict('os.environ', {}, clear=True)
    def test_missing_openai_key(self):
        from arbitration_engine.agents import call_llm_a
        with self.assertRaises(APIKeyMissingError):
            call_llm_a("test prompt")
    
    @patch.dict('os.environ', {}, clear=True)
    def test_missing_cohere_key(self):
        from arbitration_engine.agents import call_llm_b
        with self.assertRaises(APIKeyMissingError):
            call_llm_b("test prompt")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    print("="*80)
    print("Multi-LLM Arbitration Engine - Test Suite")
    print("="*80)
    print()
    
    success = run_tests()
    
    print()
    print("="*80)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("="*80)
    
    sys.exit(0 if success else 1)
