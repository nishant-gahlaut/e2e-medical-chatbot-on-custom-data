from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt, title_generation_prompt
from src.model_singletons import get_cached_llm, get_cached_embeddings
from src.pdf_loader import PyPDFLoader
import re
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import numpy as np
import time
import asyncio
from concurrent.futures import Future
from threading import Thread
import uuid
import os
import tempfile
import psutil
import logging

def log_memory_usage(step_name: str) -> None:
    """Helper function to log memory usage at a given step.
    
    Args:
        step_name: Name of the step being measured
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Convert to MB for readability
    rss_mb = memory_info.rss / 1024 / 1024
    vms_mb = memory_info.vms / 1024 / 1024
    
    # Get GPU memory if available
    gpu_memory = ""
    if torch.cuda.is_available():
        gpu_memory = f", GPU Memory: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB"
    
    print(f"🔍 Memory Usage at {step_name}:")
    print(f"   ↳ RSS: {rss_mb:.2f}MB")
    print(f"   ↳ VMS: {vms_mb:.2f}MB{gpu_memory}")

def log_time(start_time: float, step_name: str) -> float:
    """Helper function to log time taken for each step.
    
    Args:
        start_time: Start time of the step
        step_name: Name of the step being timed
        
    Returns:
        Current time for chaining measurements
    """
    current_time = time.time()
    duration = current_time - start_time
    print(f"⏱️ {step_name} took {duration:.2f} seconds")
    return current_time

def load_documents(file_path):
    """Loads a PDF document."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def run_async_in_thread(coro) -> Future:
    """Runs an async coroutine in a separate thread and returns a Future.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        Future object that will contain the result
    """
    future = Future()
    
    def _run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            loop.close()
    
    Thread(target=_run_async).start()
    return future

async def generate_document_title_async(chunks, llm) -> str:
    """Asynchronous version of document title generation.
    
    Args:
        chunks: List of document chunks
        llm: The LLM instance to use
        
    Returns:
        Generated title for the document
    """
    if not chunks or len(chunks) == 0:
        return "Untitled Document"
    
    if llm is None:
        raise ValueError("LLM instance must be provided to generate_document_title")
    
    try:
        # Take content from first 2 chunks
        sample_size = min(2, len(chunks))
        intro_text = ""
        
        for i in range(sample_size):
            if hasattr(chunks[i], 'page_content'):
                intro_text += chunks[i].page_content + " "
            elif isinstance(chunks[i], str):
                intro_text += chunks[i] + " "
                
        intro_text = intro_text[:4000].strip()
        
        if not intro_text:
            return "Untitled Document"
        
        # Create title generation prompt
        prompt = ChatPromptTemplate.from_template(title_generation_prompt)
        
        # Get title from LLM
        title_response = await llm.ainvoke(prompt.format(document_text=intro_text))
        
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

def process_file_bytes(file_bytes, filename):
    """Processes a file directly from its bytes using a temporary file.
    
    Args:
        file_bytes: The binary content of the file
        filename: Original filename (for metadata)
        
    Returns:
        Tuple of (documents, metadata) where metadata contains title and other info
    """
    total_start_time = time.time()
    print(f"\n🔄 Starting processing of file: {filename}")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        # Load documents from the temporary file
        load_start = time.time()
        documents = load_documents(temp_path)
        load_end = log_time(load_start, "PDF loading")
        
        # Create initial chunks to extract title
        chunk_start = time.time()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        initial_chunks = text_splitter.split_documents(documents)
        chunk_end = log_time(chunk_start, "Initial chunking")
        
        # Start async title generation
        title_start = time.time()
        llm = get_cached_llm()  # Use cached LLM instance
        title_future = run_async_in_thread(generate_document_title_async(initial_chunks[:2], llm))
        
        # Add initial metadata without title
        meta_start = time.time()
        temp_metadata = {
            "filename": filename,
            "page_count": len(documents),
            "timestamp": int(os.path.getmtime(temp_path)) if os.path.exists(temp_path) else None
        }
        log_time(meta_start, "Initial metadata addition")
        
        # Wait for title generation to complete
        try:
            document_title = title_future.result(timeout=10)  # 10 second timeout
        except Exception as e:
            print(f"Warning: Title generation timed out or failed: {str(e)}")
            document_title = "Untitled Document"
        
        log_time(title_start, "Async title generation")
        
        # Update metadata with title
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["title"] = document_title
        temp_metadata["title"] = document_title
        
        # Force garbage collection
        gc_start = time.time()
        gc.collect()
        log_time(gc_start, "Garbage collection")
        
        total_time = time.time() - total_start_time
        print(f"✅ Total file processing time: {total_time:.2f} seconds\n")
            
        return documents, temp_metadata
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
            log_memory_usage("Before embedding single chunk")
            embeddings_model = get_cached_embeddings()
            # Get the text content from the chunk
            text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            # Generate embedding
            embedding = embeddings_model.embed_query(text)
            log_memory_usage("After embedding single chunk")
            
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
    start_time = time.time()
    chunk_count = len(chunks)
    print(f"\n🔄 Starting parallel embedding generation for {chunk_count} chunks...")
    
    try:
        # Initialize embeddings model
        model_start = time.time()
        log_memory_usage("Before embeddings model initialization")
        _ = get_cached_embeddings()  # Warm up the model
        log_memory_usage("After embeddings model initialization")
        log_time(model_start, "Embeddings model initialization")
        
        # Generate embeddings in parallel
        embed_start = time.time()
        log_memory_usage("Before parallel embedding generation")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            embedded_chunks = list(executor.map(embed_chunk, chunks))
        log_memory_usage("After parallel embedding generation")
        log_time(embed_start, "Parallel embedding generation")
        
        # Garbage collection
        gc_start = time.time()
        log_memory_usage("Before garbage collection")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log_memory_usage("After garbage collection")
        log_time(gc_start, "Garbage collection")
        
        total_time = time.time() - start_time
        chunks_per_second = chunk_count / total_time
        print(f"✅ Embedding generation complete - {chunks_per_second:.2f} chunks/second")
        print(f"✅ Total embedding time: {total_time:.2f} seconds\n")
        
        return embedded_chunks
        
    except Exception as e:
        print(f"❌ Error in parallel embedding generation: {str(e)}")
        raise e

def create_pinecone_index(index_name):
    """Creates a Pinecone index if it does not exist."""
    load_dotenv()
    # Initialize pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        # Create index
        pc.create_index(
            name=index_name,
            dimension=1024,  # Using 768 dimensions as specified
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    return index_name

def create_pinecone_vector_store(index_name, embeddings, chunks):
    """Stores document embeddings in Pinecone and returns a retriever.
    
    Uses parallel processing for embedding generation and torch.no_grad() 
    to prevent memory accumulation.
    """
    total_start = time.time()
    print("\n🔄 Starting vector store creation...")
    
    try:
        # Generate embeddings in parallel
        embed_start = time.time()
        embedded_chunks = create_embeddings_parallel(chunks)
        log_time(embed_start, "Embedding generation")
        
        # Prepare vectors for Pinecone
        prep_start = time.time()
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
        log_time(prep_start, "Vector preparation")
        
        # Initialize Pinecone and upsert vectors
        upsert_start = time.time()
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        # Upsert in batches
        batch_size = 100
        total_vectors = len(vectors)
        for i in range(0, total_vectors, batch_size):
            batch_start = time.time()
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            batch_end = time.time()
            batch_num = (i // batch_size) + 1
            total_batches = (total_vectors + batch_size - 1) // batch_size
            print(f"  ↳ Batch {batch_num}/{total_batches} upserted in {(batch_end - batch_start):.2f} seconds")
        
        log_time(upsert_start, "Pinecone upsert")
        
        # Create retriever
        retriever_start = time.time()
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
        log_time(retriever_start, "Retriever creation")
        
        # Final garbage collection
        gc_start = time.time()
        gc.collect()
        log_time(gc_start, "Final garbage collection")
        
        total_time = time.time() - total_start
        print(f"✅ Total vector store creation time: {total_time:.2f} seconds\n")
        
        return retriever
        
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
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
        log_memory_usage("Before creating embeddings model")
        embeddings = get_cached_embeddings()
        log_memory_usage("After creating embeddings model")
        return embeddings
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise e

def get_or_create_index(index_name=None):
    """Gets existing index or creates a new one for the session.
    
    Args:
        index_name: Optional name for the index. If None, generates a new name.
    
    Returns:
        str: Name of the created or existing index
    """
    try:
        # Check if index exists
        if index_name is None:
            # Generate a session-based index name
            session_id = str(uuid.uuid4())[:8]
            index_name = f"docs-{session_id}"
            
        index_start = time.time()
        print(f"\n📊 Creating new Pinecone index: {index_name}")
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if index_name not in pc.list_indexes().names():
            index_name = create_pinecone_index(index_name)
            print(f"📊 New index created: {index_name}")
        else:
            print(f"📊 Using existing index: {index_name}")
            
        print(f"📊 Index setup complete: {time.time() - index_start:.2f} seconds")
        
        return index_name
    except Exception as e:
        print(f"⚠️ Error in index setup: {str(e)}")
        raise e
