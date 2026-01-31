# Quick Start Guide

## Installation (5 minutes)

### 1. Install Dependencies
```bash
pip install requests
```

### 2. Set Up API Keys

#### OpenAI
```bash
export OPENAI_API_KEY='sk-your-key-here'
```
Get key: https://platform.openai.com/api-keys

#### Cohere
```bash
export COHERE_API_KEY='your-key-here'
```
Get key: https://dashboard.cohere.com/api-keys

#### Ollama (Local)
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull LLaMA 3
ollama pull llama3

# Start server
ollama serve
```

### 3. Run Demo
```bash
cd arbitration_engine/pipeline
python run_pipeline.py
```

## Quick Test

```python
from arbitration_engine import run_pipeline

result = run_pipeline("What is photosynthesis?")
print(f"Confidence: {result['confidence_score']}")
print(f"Answer: {result['final_answer']}")
```

## Production Features

✓ **Error Handling**: Comprehensive exception handling with clear messages  
✓ **Logging**: Detailed logging at multiple levels  
✓ **Type Hints**: Full type annotations throughout  
✓ **Documentation**: Extensive docstrings and comments  
✓ **Testing**: Complete test suite with unittest  
✓ **Modularity**: Clean separation of concerns  
✓ **Extensibility**: Easy to add new LLMs or features  
✓ **Configuration**: Environment-based configuration  
✓ **Validation**: Input validation and sanitization  
✓ **Graceful Degradation**: Continues with available LLMs if some fail  

## Architecture Highlights

### Clean Code
- Single Responsibility Principle
- DRY (Don't Repeat Yourself)
- Meaningful variable and function names
- Consistent code style

### Production Patterns
- Factory pattern for LLM clients
- Strategy pattern for confidence scoring
- Template pattern for pipeline orchestration
- Dependency injection ready

### Error Handling
```python
try:
    result = run_pipeline(question)
except APIKeyMissingError as e:
    # Handle missing configuration
    logger.error(f"Configuration error: {e}")
except LLMAPIError as e:
    # Handle API failures
    logger.error(f"API error: {e}")
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
```

## Performance

- **3 LLM calls**: ~5-15 seconds (depending on providers)
- **Claim extraction**: <1 second
- **Agreement analysis**: <1 second
- **Total pipeline**: ~6-16 seconds

## Next Steps

1. Read the full README.md
2. Run examples.py to see different usage patterns
3. Run tests.py to verify installation
4. Customize prompts/canonical_prompt.txt for your domain
5. Add more LLM providers by extending llm_clients.py

## Support

Issues? Check the README.md troubleshooting section or run:
```bash
python tests.py  # Verify installation
python examples.py  # See usage examples
```
