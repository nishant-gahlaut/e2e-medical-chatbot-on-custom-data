from functools import lru_cache
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Global variables to store singleton instances
_llm_instance: Optional[ChatGoogleGenerativeAI] = None
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None

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

def get_cached_embeddings() -> HuggingFaceEmbeddings:
    """Returns a cached instance of the HuggingFace embeddings model."""
    global _embeddings_instance
    
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 8}
        )
    
    return _embeddings_instance

def clear_cached_models() -> None:
    """Clears all cached model instances. Useful for testing or when environment variables change."""
    global _llm_instance, _embeddings_instance
    _llm_instance = None
    _embeddings_instance = None 