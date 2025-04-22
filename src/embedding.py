from typing import List, Optional, Any
import cohere
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

class CohereEmbeddings(Embeddings):
    """Custom implementation of Cohere embeddings using direct Cohere API."""
    
    def __init__(
        self,
        cohere_api_key: Optional[str] = None,
        model_name: str = "embed-multilingual-v3.0",
        input_type: str = "search_document"
    ):
        """Initialize Cohere embeddings."""
        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                raise ValueError(
                    "COHERE_API_KEY environment variable not set. "
                    "Please set it with your Cohere API key."
                )
        
        self.client = cohere.Client(api_key=cohere_api_key)
        self.model_name = model_name
        self.input_type = input_type
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents."""
        batch_size = 96
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embed(
                texts=batch,
                model=self.model_name,
                input_type=self.input_type
            )
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text."""
        response = self.client.embed(
            texts=[text],
            model=self.model_name,
            input_type="search_query"
        )
        return response.embeddings[0]

def get_embeddings(api_key: Optional[str] = None) -> CohereEmbeddings:
    """Get Cohere cloud embeddings client."""
    return CohereEmbeddings(
        cohere_api_key=api_key,
        model_name="embed-multilingual-v3.0",
        input_type="search_document"
    ) 