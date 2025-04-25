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
    """Helper function to log memory usage at a given step."""
    process = psutil.Process()
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / 1024 / 1024
    vms_mb = memory_info.vms / 1024 / 1024

def log_time(start_time: float, step_name: str) -> float:
    """Helper function to log time taken for each step."""
    return time.time()

def load_documents(file_path):
    """Loads a PDF document."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def run_async_in_thread(coro) -> Future:
    """Runs an async coroutine in a separate thread and returns a Future."""
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
    """Asynchronous version of document title generation."""
    if not chunks or len(chunks) == 0:
        return "Untitled Document"
    
    if llm is None:
        raise ValueError("LLM instance must be provided to generate_document_title")
    
    try:
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
        
        # Add a unique identifier to force a fresh response
        prompt = ChatPromptTemplate.from_template(
            title_generation_prompt + f"\nDocument ID: {uuid.uuid4()}"
        )
        
        # Create a new event loop for this request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get title with timeout
        try:
            title_response = await asyncio.wait_for(
                llm.ainvoke(prompt.format(document_text=intro_text)),
                timeout=10.0
            )
            
            if hasattr(title_response, 'content'):
                title = title_response.content
            else:
                title = str(title_response)
            
            title = clean_title(title)
            return title if title else "Untitled Document"
            
        except asyncio.TimeoutError:
            return "Untitled Document"
        finally:
            loop.close()
            
    except Exception:
        return "Untitled Document"

def process_file_bytes(file_bytes, filename, user_id: str):
    """Processes a file directly from its bytes using a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        documents = load_documents(temp_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        initial_chunks = text_splitter.split_documents(documents)
        
        # Get a fresh LLM instance for each file
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_GEMINI_KEY"),
            convert_system_message_to_human=True
        )
        
        title_future = run_async_in_thread(generate_document_title_async(initial_chunks[:2], llm))
        
        # Generate documentId using timestamp
        document_id = f"doc_{int(time.time() * 1000)}"
        
        temp_metadata = {
            "filename": filename,
            "page_count": len(documents),
            "timestamp": int(os.path.getmtime(temp_path)) if os.path.exists(temp_path) else None,
            "user_id": user_id,
            "document_id": document_id  # Add documentId to metadata
        }
        
        try:
            document_title = title_future.result(timeout=10)
        except Exception:
            document_title = "Untitled Document"
        
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["title"] = document_title
            doc.metadata["user_id"] = user_id
            doc.metadata["document_id"] = document_id  # Add documentId to each document's metadata
        temp_metadata["title"] = document_title
        
        gc.collect()
        return documents, temp_metadata
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def create_chunk(documents):
    """Splits documents into smaller chunks with a maximum limit."""
    MAX_CHUNKS = 300
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=50)
    all_chunks = []
    
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        if len(doc_chunks) > MAX_CHUNKS:
            doc_chunks = doc_chunks[:MAX_CHUNKS]
        all_chunks.extend(doc_chunks)
    
    gc.collect()
    return all_chunks

def embed_chunk(chunk) -> Dict[str, Any]:
    """Generates embeddings for a single chunk."""
    try:
        with torch.no_grad():
            embeddings_model = get_cached_embeddings()
            text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            embedding = embeddings_model.embed_query(text)
            
            return {
                'text': text,
                'metadata': chunk.metadata if hasattr(chunk, 'metadata') else {},
                'embedding': embedding
            }
    except Exception as e:
        raise e

def create_embeddings_parallel(chunks: List, max_workers: int = None) -> List[Dict[str, Any]]:
    """Generates embeddings for chunks in parallel using ThreadPoolExecutor."""
    try:
        _ = get_cached_embeddings()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            embedded_chunks = list(executor.map(embed_chunk, chunks))
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return embedded_chunks
        
    except Exception as e:
        raise e

def create_pinecone_index(index_name):
    """Creates a Pinecone index if it does not exist."""
    load_dotenv()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    return index_name

def create_pinecone_vector_store(index_name, embeddings, chunks):
    """Stores document embeddings in Pinecone and returns a retriever."""
    try:
        embedded_chunks = create_embeddings_parallel(chunks)
        vectors = []
        
        for i, chunk_data in enumerate(embedded_chunks):
            metadata = {
                **chunk_data['metadata'],
                'text': chunk_data['text']
            }
            
            if 'user_id' not in metadata:
                if hasattr(chunks[i], 'metadata') and 'user_id' in chunks[i].metadata:
                    metadata['user_id'] = chunks[i].metadata['user_id']
                else:
                    continue
                    
            # Include documentId in metadata if available
            if hasattr(chunks[i], 'metadata') and 'document_id' in chunks[i].metadata:
                metadata['document_id'] = chunks[i].metadata['document_id']

            vectors.append({
                'id': f'chunk_{i}',
                'values': chunk_data['embedding'],
                'metadata': metadata
            })
        
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        batch_size = 100
        total_vectors = len(vectors)
        for i in range(0, total_vectors, batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3},
            search_type="similarity"
        )
        
        gc.collect()
        return retriever
        
    except Exception as e:
        raise e

def load_llm():
    """Loads the Google Gemini AI model."""
    return get_cached_llm()

def create_rag_chain(llm, retriever, system_prompt, user_input, user_id: str = None):
    """Creates the RAG pipeline to process user queries."""
    if user_id:
        if isinstance(retriever, PineconeVectorStore):
            retriever = retriever.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"user_id": user_id}
                },
                search_type="similarity"
            )
    
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
    
    title = re.sub(r'^["\'"\']|["\'"\']$', '', title.strip())
    title = re.sub(r'\s+', ' ', title)
    
    prefixes_to_remove = ['title:', 'document title:', 'suggested title:']
    for prefix in prefixes_to_remove:
        if title.lower().startswith(prefix):
            title = title[len(prefix):].strip()
    
    if not any(c.isupper() for c in title):
        title = title.title()
    
    if len(title) > 50:
        title = title[:47] + "..."
        
    return title.strip()

def create_embeddings():
    """Generates embeddings for document chunks."""
    try:
        embeddings = get_cached_embeddings()
        return embeddings
    except Exception as e:
        raise e

def get_or_create_index(index_name=None):
    """Gets existing index or creates a new one for the session."""
    try:
        if index_name is None:
            # Use a consistent index name
            index_name = "docs-index"
            
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        return index_name
    except Exception as e:
        raise e

def format_username(name: str) -> str:
    """Format username by replacing spaces with underscores and removing special characters."""
    name = ' '.join(name.split()).strip()
    name = name.replace(' ', '_')
    name = ''.join(c for c in name if c.isalnum() or c == '_')
    return name.lower()
