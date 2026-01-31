"""
LLM Client Implementations

Provides standardized interfaces to OpenAI, Cohere, and Google Gemini LLMs.
"""

import os
import json
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class APIKeyMissingError(LLMClientError):
    """Raised when required API key is not configured."""
    pass


class LLMAPIError(LLMClientError):
    """Raised when LLM API call fails."""
    pass


# ============================================================================
# OpenAI Client (LLM A)
# ============================================================================

def call_llm_a(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 1000) -> str:
    """
    Call OpenAI API (LLM A).
    
    Args:
        prompt: The prompt text to send
        model: OpenAI model identifier
        max_tokens: Maximum tokens in response
        
    Returns:
        Generated text response
        
    Raises:
        APIKeyMissingError: If OPENAI_API_KEY not set
        LLMAPIError: If API call fails
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise APIKeyMissingError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    logger.info(f"Calling OpenAI API with model: {model}")
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'choices' not in data or len(data['choices']) == 0:
            raise LLMAPIError("OpenAI API returned no choices")
        
        text = data['choices'][0]['message']['content']
        logger.info(f"OpenAI response received: {len(text)} chars")
        return text
        
    except requests.exceptions.Timeout:
        raise LLMAPIError("OpenAI API request timed out")
    except requests.exceptions.HTTPError as e:
        raise LLMAPIError(f"OpenAI API HTTP error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        raise LLMAPIError(f"OpenAI API request failed: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        raise LLMAPIError(f"Failed to parse OpenAI response: {str(e)}")


# ============================================================================
# Cohere Client (LLM B)
# ============================================================================

def call_llm_b(prompt: str, model: str = "command-a-03-2025", max_tokens: int = 1000) -> str:
    """
    Call Cohere Chat API (LLM B).
    
    Args:
        prompt: The prompt text to send
        model: Cohere model identifier
        max_tokens: Maximum tokens in response
        
    Returns:
        Generated text response
        
    Raises:
        APIKeyMissingError: If COHERE_API_KEY not set
        LLMAPIError: If API call fails
    """
    api_key = os.environ.get('COHERE_API_KEY')
    if not api_key:
        raise APIKeyMissingError(
            "COHERE_API_KEY environment variable not set. "
            "Set it with: export COHERE_API_KEY='your-key-here'"
        )
    
    logger.info(f"Calling Cohere Chat API with model: {model}")
    
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Ensure prompt is not empty
    if not prompt or len(prompt.strip()) == 0:
        raise LLMAPIError("Prompt cannot be empty")
    
    payload = {
        # "model": model,
        # "messages": [
        #     {"role": "user", "content": prompt.strip()}
        # ],
        # "max_tokens": max_tokens,
        # "temperature": 0.7
        "model": model,
    "message": prompt.strip(),
    "max_tokens": max_tokens,
    "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Handle both old and new Cohere chat formats
        if "text" in data:
            text = data["text"]
        elif "message" in data and "content" in data["message"]:
            text = "".join(
            part.get("text", "") for part in data["message"]["content"]
            if part.get("type") == "text"
            )
        else:
            raise LLMAPIError(f"Unexpected Cohere response format: {data}")

        
        text = data['text']
        logger.info(f"Cohere response received: {len(text)} chars")
        return text
        
    except requests.exceptions.Timeout:
        raise LLMAPIError("Cohere API request timed out")
    except requests.exceptions.HTTPError as e:
        raise LLMAPIError(f"Cohere API HTTP error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        raise LLMAPIError(f"Cohere API request failed: {str(e)}")
    except (KeyError, json.JSONDecodeError) as e:
        raise LLMAPIError(f"Failed to parse Cohere response: {str(e)}")


# ============================================================================
# Google Gemini Client (LLM C)
# ============================================================================

# def call_llm_c(prompt: str, model: str = "gemini-1.0-pro", max_tokens: int = 1000) -> str:
#     """
#     Call Google Gemini API (LLM C).
    
#     Args:
#         prompt: The prompt text to send
#         model: Gemini model identifier
#         max_tokens: Maximum tokens in response
        
#     Returns:
#         Generated text response
        
#     Raises:
#         APIKeyMissingError: If GEMINI_API_KEY not set
#         LLMAPIError: If API call fails
#     """
#     api_key = os.environ.get('GEMINI_API_KEY')
#     if not api_key:
#         raise APIKeyMissingError(
#             "GEMINI_API_KEY environment variable not set. "
#             "Set it with: export GEMINI_API_KEY='your-key-here'"
#         )
    
#     logger.info(f"Calling Gemini API with model: {model}")
    
#     # Use stable version for better compatibility
#     url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
#     headers = {
#         "Content-Type": "application/json"
#     }
    
#     payload = {
#         "contents": [
#             {
#                 "parts": [
#                     {"text": prompt}
#                 ]
#             }
#         ],
#         "generationConfig": {
#             "maxOutputTokens": max_tokens,
#             "temperature": 0.7
#         }
#     }
    
#     try:
#         response = requests.post(
#             f"{url}?key={api_key}",
#             headers=headers,
#             json=payload,
#             timeout=30
#         )
#         response.raise_for_status()
        
#         data = response.json()
        
#         if 'candidates' not in data or len(data['candidates']) == 0:
#             raise LLMAPIError("Gemini API returned no candidates")
        
#         if 'content' not in data['candidates'][0] or 'parts' not in data['candidates'][0]['content']:
#             raise LLMAPIError("Gemini API returned unexpected response format")
        
#         text = data['candidates'][0]['content']['parts'][0]['text']
#         logger.info(f"Gemini response received: {len(text)} chars")
#         return text
        
#     except requests.exceptions.Timeout:
#         raise LLMAPIError("Gemini API request timed out")
#     except requests.exceptions.HTTPError as e:
#         raise LLMAPIError(f"Gemini API HTTP error: {e.response.status_code} - {e.response.text}")
#     except requests.exceptions.RequestException as e:
#         raise LLMAPIError(f"Gemini API request failed: {str(e)}")
#     except (KeyError, json.JSONDecodeError) as e:
#         raise LLMAPIError(f"Failed to parse Gemini response: {str(e)}")
def call_llm_c(prompt: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 1000) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise APIKeyMissingError("GROQ_API_KEY environment variable not set")

    logger.info(f"Calling Groq API with model: {model}")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        raise LLMAPIError("Groq API request timed out")
    except requests.exceptions.HTTPError as e:
        raise LLMAPIError(f"Groq API HTTP error: {e.response.status_code} - {e.response.text}")
    except requests.exceptions.RequestException as e:
        raise LLMAPIError(f"Groq API request failed: {str(e)}")
    except (KeyError, IndexError) as e:
        raise LLMAPIError(f"Failed to parse Groq response: {str(e)}")



# ============================================================================
# Utility Functions
# ============================================================================

def test_llm_connection(llm_name: str) -> bool:
    """
    Test if an LLM client is properly configured and accessible.
    
    Args:
        llm_name: 'openai', 'cohere', or 'gemini'
        
    Returns:
        True if connection successful, False otherwise
    """
    test_prompt = "Say 'OK' if you can read this."
    
    try:
        if llm_name.lower() == 'openai':
            response = call_llm_a(test_prompt)
        elif llm_name.lower() == 'cohere':
            response = call_llm_b(test_prompt)
        elif llm_name.lower() == 'groq':
            response = call_llm_c(test_prompt)
        else:
            logger.error(f"Unknown LLM name: {llm_name}")
            return False
        
        logger.info(f"{llm_name} connection test successful")
        return True
        
    except (APIKeyMissingError, LLMAPIError) as e:
        logger.error(f"{llm_name} connection test failed: {str(e)}")
        return False
