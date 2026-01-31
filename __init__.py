"""
Multi-LLM Arbitration Engine

A production-grade research prototype for consensus-driven question answering
using multiple LLM providers with claim-level arbitration and confidence scoring.
"""

__version__ = '1.0.0'
__author__ = 'Arbitration Engine Team'

from .pipeline import run_pipeline, ArbitrationPipeline

__all__ = ['run_pipeline', 'ArbitrationPipeline']
