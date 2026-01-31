# Multi-LLM Arbitration Engine

A production-grade research prototype that orchestrates multiple Large Language Models (LLMs) to provide consensus-driven answers with explicit confidence scoring.

## Overview

This system implements a 3-LLM arbitration pipeline that:

1. **Analyzes** user questions to extract metadata (concept, intent, difficulty)
2. **Queries** three independent LLM providers with identical prompts
3. **Extracts** atomic factual claims from each response
4. **Detects** agreement and disagreement at the claim level
5. **Produces** a merged consensus answer with confidence metrics

### Key Features

- **Multi-Provider Support**: OpenAI (GPT), Cohere, and Ollama (local LLaMA 3)
- **Claim-Level Analysis**: Extracts and compares atomic factual statements
- **Confidence Scoring**: Multiple scoring algorithms (basic, weighted, entropy-based)
- **Production-Ready**: Comprehensive error handling, logging, and type hints
- **Extensible Architecture**: Modular design for easy enhancement

## Architecture

```
arbitration_engine/
├── agents/                  # LLM clients and agent implementations
│   ├── interpreter_agent.py # Question analysis and metadata extraction
│   ├── hyde_agent.py        # Hypothetical Document Embeddings generation
│   └── llm_clients.py       # OpenAI, Cohere, and Ollama API clients
├── arbitration/             # Core arbitration logic
│   ├── claim_extractor.py   # Atomic claim extraction from responses
│   ├── agreement_engine.py  # Multi-LLM consensus analysis
│   └── confidence_scorer.py # Confidence score computation
├── prompts/                 # Prompt templates
│   └── canonical_prompt.txt # Standard prompt sent to all LLMs
└── pipeline/                # Orchestration
    └── run_pipeline.py      # Main pipeline coordinator
```

## Requirements

### Python Dependencies

```bash
pip install requests
```

### LLM Provider Setup

#### 1. OpenAI API

- Sign up at: https://platform.openai.com/
- Generate API key
- Export environment variable:
  ```bash
  export OPENAI_API_KEY='sk-your-key-here'
  ```

#### 2. Cohere API

- Sign up at: https://cohere.com/
- Generate API key
- Export environment variable:
  ```bash
  export COHERE_API_KEY='your-key-here'
  ```

#### 3. Ollama (Local LLaMA 3)

- Install Ollama: https://ollama.ai/
- Pull LLaMA 3 model:
  ```bash
  ollama pull llama3
  ```
- Start Ollama server:
  ```bash
  ollama serve
  ```
- (Optional) Export custom base URL:
  ```bash
  export OLLAMA_BASE_URL='http://localhost:11434'
  ```

## Installation

1. **Clone or download the repository**

2. **Install Python dependencies**
   ```bash
   pip install requests
   ```

3. **Configure environment variables**
   ```bash
   export OPENAI_API_KEY='your-openai-key'
   export COHERE_API_KEY='your-cohere-key'
   export OLLAMA_BASE_URL='http://localhost:11434'  # Optional
   ```

4. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```

## Usage

### Quick Start

Run the demo script:

```bash
cd arbitration_engine/pipeline
python run_pipeline.py
```

The demo will:
- Check environment configuration
- Present example questions
- Allow custom question input
- Display detailed arbitration results
- Offer to save results as JSON

### Programmatic Usage

```python
from arbitration_engine.pipeline import run_pipeline

# Run arbitration on a question
result = run_pipeline("What is photosynthesis?")

# Access results
print(f"Confidence: {result['confidence_score']:.3f}")
print(f"Final Answer: {result['final_answer']}")
print(f"Disputed Claims: {len(result['disputed_claims'])}")

# Detailed analysis
for llm_name, claims in result['claim_sets'].items():
    print(f"{llm_name}: {len(claims)} claims")
```

### Advanced Usage

```python
from arbitration_engine.pipeline import ArbitrationPipeline
from arbitration_engine.agents import run_interpreter_agent

# Initialize pipeline with custom prompt
pipeline = ArbitrationPipeline(
    canonical_prompt_path='custom_prompt.txt'
)

# Run analysis
result = pipeline.run("How does quantum entanglement work?")

# Access detailed metrics
print(result['pairwise_agreement'])
print(result['statistics'])
```

## Output Structure

The pipeline returns a comprehensive dictionary:

```python
{
    'question': str,                    # Original question
    'interpreter_metadata': dict,       # Concept, intent, difficulty, keywords
    'llm_responses': dict,              # Raw responses from each LLM
    'claim_sets': dict,                 # Extracted claims per LLM
    'agreement_table': dict,            # Claims → supporting LLMs
    'supported_claims': list,           # Claims with ≥2 LLM support
    'disputed_claims': list,            # Claims with only 1 LLM support
    'unanimous_claims': list,           # Claims supported by all LLMs
    'confidence_score': float,          # Overall confidence (0-1)
    'confidence_interpretation': str,   # Human-readable confidence level
    'final_answer': str,                # Merged consensus answer
    'pairwise_agreement': dict,         # Agreement scores between LLM pairs
    'statistics': dict                  # Detailed statistics
}
```

## Confidence Scoring

The system provides multiple confidence metrics:

### Basic Confidence
Simple ratio of supported claims to total claims.

### Weighted Confidence (Primary)
Incorporates:
- **Unanimous claims** (1.5x weight): All LLMs agree
- **Supported claims** (1.0x weight): ≥2 LLMs agree
- **Disputed claims** (0.3x penalty): Only 1 LLM supports

### Entropy-Based Confidence
Uses information entropy of agreement distribution:
- Lower entropy (more consensus) → Higher confidence
- Higher entropy (more disagreement) → Lower confidence

### Interpretation Levels
- **≥0.90**: Very High - Strong consensus
- **≥0.75**: High - Majority agreement
- **≥0.60**: Moderate - Reasonable consensus
- **≥0.40**: Low - Significant disagreement
- **<0.40**: Very Low - Little consensus

## Claim Extraction

Claims are extracted using:
1. **Sentence segmentation**: Split responses into sentences
2. **Factual filtering**: Remove non-factual content (opinions, hedges)
3. **Normalization**: Lowercase, whitespace cleanup, punctuation removal
4. **Atomic claims**: Each claim is a single verifiable statement

Example:
```
Response: "Photosynthesis is the process by which plants convert light into energy. It occurs in chloroplasts."

Claims:
- "photosynthesis is the process by which plants convert light into energy"
- "it occurs in chloroplasts"
```

## Agreement Analysis

### Agreement Levels
- **Unanimous (3/3)**: All LLMs agree
- **Supported (≥2/3)**: Consensus threshold met
- **Disputed (1/3)**: Single LLM only

### Pairwise Agreement
Jaccard similarity between each LLM pair:
```
similarity = |claims₁ ∩ claims₂| / |claims₁ ∪ claims₂|
```

## Error Handling

The system includes comprehensive error handling:

- **Missing API Keys**: Clear error messages with setup instructions
- **LLM Failures**: Graceful degradation (continues with available LLMs)
- **Network Errors**: Timeout handling and retry suggestions
- **Invalid Responses**: Validation and fallback mechanisms

## Logging

Detailed logging at multiple levels:

```python
import logging

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Logs include:
# - API calls and response sizes
# - Claim extraction counts
# - Agreement statistics
# - Confidence calculations
# - Error traces
```

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

### "Cannot connect to Ollama"
Ensure Ollama is running:
```bash
ollama serve
```

### "All LLM calls failed"
Check:
1. API keys are correctly set
2. Network connectivity
3. API rate limits
4. Ollama server is running (for Ollama)

### Low confidence scores
This is normal when:
- Question is ambiguous
- LLMs interpret question differently
- Topic has multiple valid perspectives

## Limitations

- **No retry logic**: Single API call per LLM (by design)
- **Synchronous execution**: LLMs called sequentially
- **Simple claim extraction**: Uses sentence-based splitting
- **No caching**: Each run makes fresh API calls
- **English only**: Optimized for English language queries

## Future Enhancements

Potential improvements for production deployment:

1. **Async execution**: Parallel LLM calls
2. **Advanced NLP**: Semantic claim similarity
3. **Caching layer**: Response and claim caching
4. **Retry logic**: Exponential backoff for failures
5. **More LLMs**: Anthropic Claude, Google PaLM, etc.
6. **Web interface**: Dashboard for results visualization
7. **Database storage**: Persistent result storage
8. **A/B testing**: Prompt optimization framework

## License

Research prototype - use as needed for research and educational purposes.

## Contributing

This is a research prototype. Feel free to extend and modify for your use case.

## Support

For issues with:
- **OpenAI API**: https://platform.openai.com/docs
- **Cohere API**: https://docs.cohere.com/
- **Ollama**: https://github.com/ollama/ollama

---

**Note**: This system is designed for research and prototyping. Production deployment would require additional security, monitoring, and optimization.
