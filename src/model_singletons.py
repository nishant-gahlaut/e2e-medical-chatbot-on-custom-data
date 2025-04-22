from functools import lru_cache
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from src.embedding import get_embeddings, CohereEmbeddings

# Global variables to store singleton instances
_llm_instance: Optional[ChatGoogleGenerativeAI] = None
_embeddings_instance: Optional[CohereEmbeddings] = None

def get_cached_llm() -> ChatGoogleGenerativeAI:
    """Returns a cached instance of the Google Gemini AI model."""
    global _llm_instance
    
    if _llm_instance is None:
        api_key = os.getenv("GOOGLE_GEMINI_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_GEMINI_KEY environment variable not set. "
                "Please set it with your Google API key."
            )
        
        _llm_instance = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            convert_system_message_to_human=True
        )
    
    return _llm_instance

def get_cached_embeddings() -> CohereEmbeddings:
    """Returns a cached instance of the Cohere embeddings model."""
    global _embeddings_instance
    
    if _embeddings_instance is None:
        _embeddings_instance = get_embeddings()
    
    return _embeddings_instance

def clear_cached_models() -> None:
    """Clears all cached model instances."""
    global _llm_instance, _embeddings_instance
    _llm_instance = None
    _embeddings_instance = None 