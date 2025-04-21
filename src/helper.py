from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt, title_generation_prompt
from src.model_singletons import get_cached_llm, get_cached_embeddings
import re
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np

def load_documents(file_path):
    """Loads a PDF document."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def process_file_bytes(file_bytes, filename):
    """Processes a file directly from its bytes using a temporary file.
    
    Args:
        file_bytes: The binary content of the file
        filename: Original filename (for metadata)
        
    Returns:
        Tuple of (documents, metadata) where metadata contains title and other info
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        # Load documents from the temporary file
        print(f"Processing file: {filename}")
        documents = load_documents(temp_path)
        
        # Create initial chunks to extract title
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        initial_chunks = text_splitter.split_documents(documents)
        
        # Load LLM once and reuse
        llm = load_llm()
        document_title = generate_document_title(initial_chunks[:2], llm=llm)  # Explicitly pass llm
        
        # Add metadata to documents
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["title"] = document_title
        
        # Create file metadata
        file_metadata = {
            "filename": filename,
            "title": document_title,
            "page_count": len(documents),
            "timestamp": int(os.path.getmtime(temp_path)) if os.path.exists(temp_path) else None
        }
        
        # Force garbage collection after processing large file
        gc.collect()
            
        return documents, file_metadata
    finally:
        # Always clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def create_chunk(documents):
    """Splits documents into smaller chunks with a maximum limit.
    
    Args:
        documents: List of documents to split
        
    Returns:
        List of chunks, limited to MAX_CHUNKS per document to control memory usage
    """
    MAX_CHUNKS = 300  # Maximum chunks per document to control memory and API usage
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []
    
    for doc in documents:
        # Split individual document
        doc_chunks = text_splitter.split_documents([doc])
        
        # Limit chunks for this document
        if len(doc_chunks) > MAX_CHUNKS:
            print(f"Warning: Document '{doc.metadata.get('source', 'unknown')}' exceeded chunk limit. "
                  f"Using first {MAX_CHUNKS} chunks out of {len(doc_chunks)}")
            doc_chunks = doc_chunks[:MAX_CHUNKS]
        
        all_chunks.extend(doc_chunks)
    
    print(f"Total chunks after limiting: {len(all_chunks)}")
    
    # Force garbage collection after creating chunks
    gc.collect()
    
    return all_chunks

def embed_chunk(chunk) -> Dict[str, Any]:
    """Generates embeddings for a single chunk.
    
    Args:
        chunk: Document chunk to embed
        
    Returns:
        Dictionary containing the chunk's text, metadata, and embedding
    """
    try:
        with torch.no_grad():
            embeddings_model = get_cached_embeddings()
            # Get the text content from the chunk
            text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            # Generate embedding
            embedding = embeddings_model.embed_query(text)
            
            return {
                'text': text,
                'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {},
                'embedding': embedding
            }
    except Exception as e:
        print(f"Error embedding chunk: {str(e)}")
        raise e

def create_embeddings_parallel(chunks: List, max_workers: int = None) -> List[Dict[str, Any]]:
    """Generates embeddings for chunks in parallel using ThreadPoolExecutor.
    
    Args:
        chunks: List of document chunks to embed
        max_workers: Maximum number of worker threads (None for default based on CPU count)
        
    Returns:
        List of dictionaries containing text, metadata, and embeddings
    """
    print(f"Generating embeddings for {len(chunks)} chunks in parallel...")
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map the embedding function across all chunks in parallel
            embedded_chunks = list(executor.map(embed_chunk, chunks))
            
        # Force garbage collection after parallel embedding
        gc.collect()
        
        print(f"Successfully generated {len(embedded_chunks)} embeddings")
        return embedded_chunks
        
    except Exception as e:
        print(f"Error in parallel embedding generation: {str(e)}")
        raise e

def create_pinecone_index(index_name):
    """Creates a Pinecone index if it does not exist."""
    load_dotenv()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = index_name
    
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return index_name

def create_pinecone_vector_store(index_name, embeddings, chunks):
    """Stores document embeddings in Pinecone and returns a retriever.
    
    Uses parallel processing for embedding generation and torch.no_grad() 
    to prevent memory accumulation.
    """
    try:
        # Generate embeddings in parallel
        embedded_chunks = create_embeddings_parallel(chunks)
        
        # Prepare data for Pinecone
        vectors = []
        for i, chunk_data in enumerate(embedded_chunks):
            vectors.append({
                'id': f'chunk_{i}',
                'values': chunk_data['embedding'],
                'metadata': {
                    **chunk_data['metadata'],
                    'text': chunk_data['text']
                }
            })
        
        # Initialize Pinecone and upsert vectors
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        # Upsert in batches to avoid overwhelming the API
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        # Create retriever
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
        
        # Force garbage collection after embedding operations
        gc.collect()
        
        return retriever
        
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise e

def load_llm():
    """Loads the Google Gemini AI model."""
    return get_cached_llm()

def create_rag_chain(llm, retriever, system_prompt, user_input):
    """Creates the RAG pipeline to process user queries."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_input),
    ])
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )
    response = rag_chain.invoke({"input": user_input})
    return response["answer"]

def generate_document_title(chunks, llm):
    """Generates a title for a document using an LLM based on the first few chunks.
    
    Args:
        chunks: List of document chunks, with the first few representing the document intro
        llm: The LLM instance to use (required) - reuses existing instance to avoid reloading
        
    Returns:
        A generated title for the document
    """
    if not chunks or len(chunks) == 0:
        return "Untitled Document"
    
    if llm is None:
        raise ValueError("LLM instance must be provided to generate_document_title")
    
    try:
        # Take content from first 2 chunks (or fewer if not available)
        sample_size = min(2, len(chunks))
        intro_text = ""
        
        for i in range(sample_size):
            if hasattr(chunks[i], 'page_content'):
                intro_text += chunks[i].page_content + " "
            elif isinstance(chunks[i], str):
                intro_text += chunks[i] + " "
                
        # Limit text length (approximately 1000 tokens)
        intro_text = intro_text[:4000].strip()
        
        if not intro_text:
            return "Untitled Document"
        
        # Create title generation prompt
        prompt = ChatPromptTemplate.from_template(title_generation_prompt)
        
        # Get title from LLM
        title_response = llm.invoke(prompt.format(document_text=intro_text))
        
        # Extract response
        if hasattr(title_response, 'content'):
            title = title_response.content
        else:
            title = str(title_response)
        
        # Clean up title
        title = clean_title(title)
        print(f"Generated title: {title}")
        
        return title if title else "Untitled Document"
        
    except Exception as e:
        print(f"Error generating title: {str(e)}")
        return "Untitled Document"

def clean_title(title):
    """Cleans up a generated title by removing quotes, extra whitespace, etc."""
    if not title:
        return ""
    
    # Remove quotes and extra spaces
    title = re.sub(r'^["\'"\']|["\'"\']$', '', title.strip())
    title = re.sub(r'\s+', ' ', title)
    
    # Remove common prefixes LLMs might add
    prefixes_to_remove = ['title:', 'document title:', 'suggested title:']
    for prefix in prefixes_to_remove:
        if title.lower().startswith(prefix):
            title = title[len(prefix):].strip()
    
    # Ensure proper capitalization (if title is too lowercase)
    if not any(c.isupper() for c in title):
        title = title.title()
    
    # Limit length
    if len(title) > 50:
        title = title[:47] + "..."
        
    return title.strip()

def create_embeddings():
    """Generates embeddings for document chunks.
    
    This function maintains backward compatibility while using the new parallel implementation internally.
    Returns the embeddings model instance.
    """
    try:
        return get_cached_embeddings()
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise e
