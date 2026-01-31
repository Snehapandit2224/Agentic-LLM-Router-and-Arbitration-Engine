from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

from agents.llm_clients import call_llm_b, call_llm_c

print("COHERE:", call_llm_b("Say OK"))
print("GROQ:", call_llm_c("Say OK"))
