import os
from src.helper import (
    load_documents, create_chunk, create_embeddings,
    create_pinecone_index, create_pinecone_vector_store, create_rag_chain,
    get_or_create_index
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.model_singletons import get_cached_llm, get_cached_embeddings
from src.prompt import system_prompt

def process_documents():
    """Loads and processes all uploaded documents."""
    upload_dir = "uploads/"
    file_paths = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith(".pdf")]
    
    if not file_paths:
        return "No files found for processing."
    
    all_chunks = []
    for file_path in file_paths:
        documents = load_documents(file_path)
        chunks = create_chunk(documents)
        all_chunks.extend(chunks)
    
    embeddings = create_embeddings(all_chunks)
    index_name = create_pinecone_index()
    vector_store = create_pinecone_vector_store(index_name, embeddings)
    
    return vector_store

def get_response(query, llm, vector_store):
    """
    Get a response for a user query using the provided LLM and vector store.
    
    Args:
        query (str): The user's query
        llm: The language model to use
        vector_store: The vector store to use for retrieval
        
    Returns:
        str: The response to the user's query
    """
    try:
        response = create_rag_chain(llm, vector_store, system_prompt, query)
        return response
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return f"I apologize, but I encountered an error: {str(e)}"

def process_file(file_path, index_name):
    """Process a file and store its embeddings in Pinecone."""
    try:
        # Load and chunk documents
        documents = load_documents(file_path)
        chunks = create_chunk(documents)
        
        # Create embeddings
        embeddings = create_embeddings()
        
        # Initialize Pinecone and create index
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = get_or_create_index(index_name)
        
        # Create vector store
        vector_store = create_pinecone_vector_store(index_name, embeddings, chunks)
        
        return True, "File processed successfully"
    except Exception as e:
        return False, str(e)
