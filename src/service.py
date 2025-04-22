import os
import gc
import torch
from typing import List, Dict, Tuple, Any
from werkzeug.datastructures import FileStorage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_documents,
    create_chunk,
    create_embeddings,
    create_pinecone_vector_store,
    create_rag_chain,
    get_or_create_index,
    process_file_bytes
)
from src.model_singletons import (
    get_cached_llm,
    get_cached_embeddings
)
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
    index_name = get_or_create_index()
    vector_store = create_pinecone_vector_store(index_name, embeddings)
    
    return vector_store

def get_response(query, llm, vector_store):
    """Get a response for a user query using the provided LLM and vector store."""
    try:
        response = create_rag_chain(llm, vector_store, system_prompt, query)
        return response
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

def process_file(file_path, index_name):
    """Process a file and store its embeddings in Pinecone."""
    try:
        documents = load_documents(file_path)
        chunks = create_chunk(documents)
        embeddings = create_embeddings()
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = get_or_create_index(index_name)
        vector_store = create_pinecone_vector_store(index_name, embeddings, chunks)
        return True, "File processed successfully"
    except Exception as e:
        return False, str(e)

def process_uploaded_files(
    files: List[FileStorage],
    user_id: str,
    max_file_size: int = 10 * 1024 * 1024
) -> Tuple[Dict[str, Any], bool, str]:
    """Process uploaded files and store them in the vector database."""
    try:
        all_documents = []
        all_metadata = []
        processed_files = []
        
        for file in files:
            if file.filename:
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > max_file_size:
                    return {}, False, f"File {file.filename} exceeds the {max_file_size/1024/1024}MB size limit"
                
                file_bytes = file.read()
                documents, file_metadata = process_file_bytes(file_bytes, file.filename, user_id)
                
                all_documents.extend(documents)
                all_metadata.append(file_metadata)
                processed_files.append(file.filename)
        
        _documents = all_documents
        _chunks = create_chunk(_documents)
        
        BATCH_SIZE = 50
        all_batches = [_chunks[i:i + BATCH_SIZE] for i in range(0, len(_chunks), BATCH_SIZE)]
        
        _embeddings = create_embeddings()
        
        # Get the consistent index name from get_or_create_index
        index_name = "docs-index"
        # index_name = get_or_create_index()
        if not index_name:
            return {}, False, "Index not initialized yet. Please wait a moment and try again."
        
        for i, batch in enumerate(all_batches):
            if i == 0:
                _vector_store = create_pinecone_vector_store(index_name, _embeddings, batch)
            else:
                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=_embeddings,
                    index_name=index_name,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY")
                )
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "message": f"Successfully processed {len(processed_files)} files",
            "documents": all_metadata,
            "document_count": len(all_documents),
            "chunk_count": len(_chunks),
            "index_name": index_name
        }, True, ""
        
    except Exception as e:
        return {}, False, str(e)

def delete_user_data(index_name: str, user_id: str, document_ids: List[str]) -> Tuple[bool, str]:
    """Delete specific documents from the current session in Pinecone index.
    
    Args:
        index_name: Name of the Pinecone index
        user_id: ID of the user whose data should be deleted
        document_ids: List of document IDs from current session's _document_metadata
        
    Returns:
        Tuple of (success, message)
    """
    try:
        if not document_ids:
            return False, "No documents to delete in current session"

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        # Delete only documents that match both user_id and document_ids from current session
        for doc_id in document_ids:
            index.delete(filter={
                "user_id": user_id,
                "document_id": doc_id
            })
        
        return True, f"Successfully deleted {len(document_ids)} documents from current session"
            
    except Exception as e:
        return False, str(e)

def process_search_query(query: str, user_id: str, index_name: str) -> Tuple[str, bool, str]:
    """Process a search query and return the response.
    
    Args:
        query: The user's search query
        user_id: The ID of the user making the query
        index_name: Name of the Pinecone index to search in
        
    Returns:
        Tuple of (response, success, error_message)
    """
    try:
        if not query:
            return "", False, "Query parameter is required"
            
        if not index_name:
            return "", False, "No index available"
            
        # Create embeddings instance
        embeddings = get_cached_embeddings()
        
        # Create vector store with user filtering
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Set up retriever with user filtering
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"user_id": user_id}  # Filter by user_id
            },
            search_type="similarity"
        )
        
        # Get LLM instance
        llm = get_cached_llm()
        
        # Get response using RAG chain
        response = create_rag_chain(
            llm=llm,
            retriever=retriever,
            system_prompt=system_prompt,
            user_input=query,
            user_id=user_id
        )
        
        return response, True, ""
        
    except Exception as e:
        return "", False, str(e)
