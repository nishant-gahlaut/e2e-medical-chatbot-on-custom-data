import os
import time
import threading
from flask import Flask, request, jsonify, render_template, url_for, session, redirect
from dotenv import load_dotenv
from src.service import get_response
import shutil
from src.helper import (
    load_documents, process_file_bytes, create_chunk, create_embeddings,
    create_pinecone_index, create_pinecone_vector_store, create_rag_chain,
    get_or_create_index, log_memory_usage, format_username
)
import psutil
import torch
from src.prompt import system_prompt
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import uuid
from src.model_singletons import get_cached_llm, get_cached_embeddings
import gc

load_dotenv()

def log_app_memory(step_name: str):
    """Log memory usage with additional process information."""
    process = psutil.Process()
    
    # CPU Usage
    cpu_percent = process.cpu_percent()
    
    # Memory details
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / 1024 / 1024
    vms_mb = memory_info.vms / 1024 / 1024
    
    # System memory
    system_memory = psutil.virtual_memory()
    system_memory_used_percent = system_memory.percent
    
    print(f"\n📊 System Resources at {step_name}:")
    print(f"   ↳ Process RSS Memory: {rss_mb:.2f}MB")
    print(f"   ↳ Process VMS Memory: {vms_mb:.2f}MB")
    print(f"   ↳ CPU Usage: {cpu_percent}%")
    print(f"   ↳ System Memory Usage: {system_memory_used_percent}%")
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"   ↳ GPU Memory Allocated: {gpu_memory_allocated:.2f}MB")
        print(f"   ↳ GPU Memory Reserved: {gpu_memory_reserved:.2f}MB")

# Log initial memory usage when app starts
print("\n🚀 Starting Flask application...")
log_app_memory("Application Startup")

app = Flask(__name__, 
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

# Log memory after Flask app creation
log_app_memory("After Flask App Creation")

# Set a secret key for session management
app.secret_key = os.urandom(24)
UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_GEMINI_KEY"] = GOOGLE_GEMINI_KEY

# Global variables to store initialized components
_documents = None
_chunks = None
_embeddings = None
_vector_store = None
_index_name = None
_llm = None
_document_metadata = []  # List to store document metadata

def get_index_name():
    """Get the current index name from app config."""
    return app.config.get('INDEX_NAME')

def set_index_name(name):
    """Set the index name in app config."""
    app.config['INDEX_NAME'] = name

def initialize_models():
    """Initialize all models in background."""
    try:
        # Initialize LLM
        print("\n🤖 Initializing LLM model...")
        llm_start = time.time()
        _llm = get_cached_llm()
        print(f"✅ LLM initialized: {time.time() - llm_start:.2f} seconds")
        
        # Initialize embeddings
        print("\n📊 Initializing embeddings model...")
        embed_start = time.time()
        _embeddings = create_embeddings()
        print(f"✅ Embeddings initialized: {time.time() - embed_start:.2f} seconds")
        
        print("\n✨ All models initialized successfully")
    except Exception as e:
        print(f"⚠️ Error initializing models: {str(e)}")

@app.route('/')
def index():
    """Render landing page."""
    # Check if user is already logged in
    if 'user_id' in session:
        return redirect(url_for('chat'))
        
    # Log memory before initialization
    log_app_memory("Before / Endpoint Processing")
    
    # Initialize index and models in background
    if not get_index_name():
        threading.Thread(target=lambda: set_index_name(get_or_create_index(get_index_name()))).start()
    threading.Thread(target=initialize_models).start()
    
    # Log memory after starting initialization threads
    log_app_memory("After Starting Initialization Threads")
    
    return render_template('landing.html')

@app.route('/chat')
def chat():
    """Render chat UI."""
    # Ensure user has entered their name
    if 'user_id' not in session:
        return redirect(url_for('index'))
    
    return render_template('chat.html')

@app.route('/set_name', methods=['POST'])
def set_name():
    """Store user's name in session."""
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    
    # Format the username and use it directly as user_id
    user_id = format_username(name)
    if not user_id:
        return jsonify({"success": False, "error": "Invalid name format"}), 400
    
    # Store formatted name only
    session['user_id'] = user_id
    
    print(f"User registered with ID: '{user_id}'")
    return jsonify({"success": True, "redirect": url_for('chat')})

@app.route('/get_user_name')
def get_user_name():
    """Get user's ID from session."""
    return jsonify({"name": session.get('user_id', '')})

@app.route('/upload', methods=['POST'])
def upload():
    """Handles multiple file uploads and processes them for vector storage without permanent storage."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _document_metadata
    
    # Get user ID from session
    user_id = session.get('user_id', 'anonymous')
    
    request_start_time = time.time()
    print(f"\n🔄 Starting file upload request processing for user: {user_id}...")
    log_app_memory("Start of Upload Request")
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    
    try:
        # Process files directly without saving to permanent storage
        all_documents = []
        all_metadata = []
        processed_files = []
        
        # Maximum file size (10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024
        
        # File Processing Phase
        file_processing_start = time.time()
        print("\n📁 Starting file processing phase...")
        log_app_memory("Before File Processing")
        
        for file in files:
            if file.filename:
                file_start_time = time.time()
                print(f"\n  Processing file: {file.filename}")
                
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > MAX_FILE_SIZE:
                    return jsonify({"error": f"File {file.filename} exceeds the 10MB size limit"}), 400
                
                # Read file bytes
                read_start = time.time()
                file_bytes = file.read()
                print(f"  ⏱️ File read took {time.time() - read_start:.2f} seconds")
                
                # Process the file bytes directly using a temporary file
                process_start = time.time()
                documents, file_metadata = process_file_bytes(file_bytes, file.filename, user_id)
                print(f"  ⏱️ File processing took {time.time() - process_start:.2f} seconds")
                
                all_documents.extend(documents)
                all_metadata.append(file_metadata)
                processed_files.append(file.filename)
                
                print(f"  ✅ Total time for {file.filename}: {time.time() - file_start_time:.2f} seconds")
                log_app_memory(f"After Processing {file.filename}")
        
        # Store document metadata
        _document_metadata = all_metadata
        print(f"📊 Documents processed: {len(_document_metadata)}")
        log_app_memory("After All File Processing")
        
        # Chunking Phase
        chunk_start = time.time()
        print("\n🔪 Starting chunking phase...")
        log_app_memory("Before Chunking")
        _documents = all_documents
        _chunks = create_chunk(_documents)
        chunk_time = time.time() - chunk_start
        print(f"🔪 Chunking complete: {chunk_time:.2f} seconds, {len(_chunks)} chunks created")
        log_app_memory("After Chunking")
        
        # Process in smaller batches if many chunks
        BATCH_SIZE = 50
        all_batches = [_chunks[i:i + BATCH_SIZE] for i in range(0, len(_chunks), BATCH_SIZE)]
        print(f"📦 Created {len(all_batches)} batch(es) of maximum size {BATCH_SIZE}")
        
        # Get existing embeddings model
        embed_start = time.time()
        print("\n🔤 Getting embeddings model...")
        log_app_memory("Before Loading Embeddings Model")
        _embeddings = create_embeddings()
        print(f"🔤 Got embeddings model: {time.time() - embed_start:.2f} seconds")
        log_app_memory("After Loading Embeddings Model")
        
        # Get existing index
        index_start = time.time()
        index_name = get_index_name()
        if not index_name:
            return jsonify({"error": "Index not initialized yet. Please wait a moment and try again."}), 503
        print(f"📊 Got index: {time.time() - index_start:.2f} seconds")
        
        # Vector store creation phase
        vector_store_start = time.time()
        print("\n💾 Starting vector store creation...")
        log_app_memory("Before Vector Store Creation")
        
        for i, batch in enumerate(all_batches):
            batch_start = time.time()
            print(f"\n  Processing batch {i+1}/{len(all_batches)} ({len(batch)} chunks)")
            log_app_memory(f"Before Processing Batch {i+1}")
            
            if i == 0:
                # First batch creates the store
                _vector_store = create_pinecone_vector_store(index_name, _embeddings, batch)
            else:
                # Add subsequent batches
                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=_embeddings,
                    index_name=index_name,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY")
                )
            print(f"  ✅ Batch {i+1} complete: {time.time() - batch_start:.2f} seconds")
            log_app_memory(f"After Processing Batch {i+1}")
        
        vector_store_time = time.time() - vector_store_start
        print(f"💾 Vector store creation complete: {vector_store_time:.2f} seconds\n")
        log_app_memory("After Vector Store Creation")
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log_app_memory("After Garbage Collection")
        
        total_time = time.time() - request_start_time
        print(f"\n✨ Total upload request processing time: {total_time:.2f} seconds")
        
        return jsonify({
            "message": "Files processed successfully", 
            "processed_files": processed_files,
            "document_count": len(_documents),
            "chunk_count": len(_chunks),
            "documents": _document_metadata,
            "processing_time": {
                "file_processing": file_processing_start,
                "chunking": chunk_time,
                "vector_store_creation": vector_store_time,
                "total": total_time
            },
            "index_name": index_name
        })
    except Exception as e:
        print(f"❌ Error processing files: {str(e)}")
        log_app_memory("After Error")
        return jsonify({"error": str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """Returns the list of uploaded documents with metadata."""
    global _document_metadata
    
    # Simplified log
    if _document_metadata:
        print(f"Returning {len(_document_metadata)} documents")
    
    if not _document_metadata:
        return jsonify({"documents": [], "message": "No documents uploaded yet"})
    
    return jsonify({"documents": _document_metadata})

@app.route("/search", methods=["GET"])
def chatbot():
    """Handles user queries by fetching relevant information."""
    global _documents, _chunks, _embeddings, _vector_store, _llm
    
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        # Check if components are initialized
        if _vector_store is None:
            return jsonify({"error": "Please upload documents first"}), 400
        
        # Initialize LLM if not already done
        if _llm is None:
            _llm = get_cached_llm()
            
        # Ensure we're using the same embeddings instance
        if _embeddings is None:
            _embeddings = get_cached_embeddings()
        
        # Get user ID from session for filtering
        user_id = session.get('user_id', 'anonymous')
        print(f"\n🔍 Processing query for user: {user_id}")
        
        # Use the stored components to get a response with user filtering
        response = create_rag_chain(
            llm=_llm,
            retriever=_vector_store,
            system_prompt=system_prompt,
            user_input=query,
            user_id=user_id
        )
        
        return jsonify({
            "response": response,
            "user_context": user_id  # Include user context in response
        })
    except Exception as e:
        print(f"Error in chatbot endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Clears uploaded documents from the library."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _document_metadata
    
    try:
        # Reset document-related variables
        _documents = None
        _chunks = None
        _embeddings = None
        _vector_store = None
        _document_metadata = []
        
        # Force garbage collection
        gc.collect()
        
        print("✨ Document library cleared successfully")
        return jsonify({
            "message": "Document library cleared successfully",
            "status": "success"
        })
    except Exception as e:
        print(f"❌ Error during document reset: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Deletes Pinecone index and resets the entire session."""
    try:
        # Delete Pinecone index if it exists
        index_name = get_index_name()
        if index_name:
            try:
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                pc.delete_index(index_name)
                print(f"🗑️ Deleted Pinecone index: {index_name}")
                # Clear the index name from config
                set_index_name(None)
            except Exception as e:
                print(f"⚠️ Error deleting Pinecone index: {str(e)}")
        
        # Call the regular reset to clear documents
        reset()
        
        print("✨ Session reset successfully")
        return jsonify({
            "message": "Session reset successfully",
            "status": "success"
        })
    except Exception as e:
        print(f"❌ Error during session reset: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Set higher timeouts to allow for processing larger files
    from werkzeug.serving import run_simple
    
    # Configure app for production-like settings
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload
    
    run_simple('0.0.0.0', 8080, app, 
               use_reloader=True,
               use_debugger=True,
               threaded=True,
               passthrough_errors=True)
