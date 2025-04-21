import os
import time
from flask import Flask, request, jsonify, render_template, url_for, session
from dotenv import load_dotenv
from src.service import get_response
import shutil
from src.helper import load_documents, process_file_bytes, create_chunk, create_embeddings, create_pinecone_index, create_pinecone_vector_store, create_rag_chain
from src.prompt import system_prompt
from pinecone import Pinecone
import uuid
from src.model_singletons import get_cached_llm

load_dotenv()

app = Flask(__name__, 
            static_url_path='', 
            static_folder='static',
            template_folder='templates')
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

def get_or_create_index():
    """Gets existing index or creates a new one for the session."""
    global _index_name
    try:
        # Check if index exists in session
        if _index_name is None:
            # Generate a session-based index name
            session_id = str(uuid.uuid4())[:8]
            index_name = f"docs-{session_id}"
            
            index_start = time.time()
            print(f"\n📊 Creating new Pinecone index: {index_name}")
            
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            if index_name not in pc.list_indexes():
                index_name = create_pinecone_index(index_name)
                print(f"📊 New index created: {index_name}")
            else:
                print(f"📊 Using existing index: {index_name}")
            
            # Store in session
            _index_name = index_name
            print(f"📊 Index setup complete: {time.time() - index_start:.2f} seconds")
        else:
            index_name=_index_name
            print(f"📊 Reusing existing index from session: {index_name}")
        
        return index_name
    except Exception as e:
        print(f"⚠️ Error in index setup: {str(e)}")
        raise e

@app.route('/')
def index():
    """Render chat UI."""
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handles multiple file uploads and processes them for vector storage without permanent storage."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _document_metadata
    
    request_start_time = time.time()
    print("\n🔄 Starting file upload request processing...")
    
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
                documents, file_metadata = process_file_bytes(file_bytes, file.filename)
                print(f"  ⏱️ File processing took {time.time() - process_start:.2f} seconds")
                
                all_documents.extend(documents)
                all_metadata.append(file_metadata)
                processed_files.append(file.filename)
                
                print(f"  ✅ Total time for {file.filename}: {time.time() - file_start_time:.2f} seconds")
        
        file_processing_time = time.time() - file_processing_start
        print(f"📁 File processing phase complete: {file_processing_time:.2f} seconds")
        
        # Store document metadata
        _document_metadata = all_metadata
        print(f"📊 Documents processed: {len(_document_metadata)}")
        
        # Chunking Phase
        chunk_start = time.time()
        print("\n🔪 Starting chunking phase...")
        _documents = all_documents
        _chunks = create_chunk(_documents)
        chunk_time = time.time() - chunk_start
        print(f"🔪 Chunking complete: {chunk_time:.2f} seconds, {len(_chunks)} chunks created")
        
        # Process in smaller batches if many chunks
        BATCH_SIZE = 50
        all_batches = [_chunks[i:i + BATCH_SIZE] for i in range(0, len(_chunks), BATCH_SIZE)]
        print(f"📦 Created {len(all_batches)} batch(es) of maximum size {BATCH_SIZE}")
        
        # Initialize embeddings
        embed_start = time.time()
        print("\n🔤 Initializing embeddings model...")
        _embeddings = create_embeddings()
        print(f"🔤 Embeddings model initialized: {time.time() - embed_start:.2f} seconds")
        
        # Get or create Pinecone index (reuse existing if available)
        index_start = time.time()
        index_name = get_or_create_index()
        print(f"📊 Index operation took: {time.time() - index_start:.2f} seconds")
        
        # Vector store creation phase
        vector_store_start = time.time()
        print("\n💾 Starting vector store creation...")
        
        for i, batch in enumerate(all_batches):
            batch_start = time.time()
            print(f"\n  Processing batch {i+1}/{len(all_batches)} ({len(batch)} chunks)")
            
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
        
        vector_store_time = time.time() - vector_store_start
        print(f"💾 Vector store creation complete: {vector_store_time:.2f} seconds")
        
        total_time = time.time() - request_start_time
        print(f"\n✨ Total upload request processing time: {total_time:.2f} seconds")
        
        return jsonify({
            "message": "Files processed successfully", 
            "processed_files": processed_files,
            "document_count": len(_documents),
            "chunk_count": len(_chunks),
            "documents": _document_metadata,
            "processing_time": {
                "file_processing": file_processing_time,
                "chunking": chunk_time,
                "vector_store_creation": vector_store_time,
                "total": total_time
            },
            "index_name": index_name  # Return index name for verification
        })
    except Exception as e:
        print(f"❌ Error processing files: {str(e)}")
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
        
        # Use the stored components to get a response
        response = create_rag_chain(_llm, _vector_store, system_prompt, query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in chatbot endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Deletes Pinecone indexes and resets server-side resources."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _index_name, _document_metadata
    
    try:
        # Delete Pinecone index if it exists
        if _index_name:
            try:
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                pc.delete_index(_index_name)
                print(f"🗑️ Deleted Pinecone index: {_index_name}")
            except Exception as e:
                print(f"⚠️ Error deleting Pinecone index: {str(e)}")
        
        # Reset all global variables
        _documents = None
        _chunks = None
        _embeddings = None
        _vector_store = None
        _index_name = None
        _document_metadata = []
        
        # Explicitly reset LLM to ensure fresh initialization
        _llm = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("✨ All resources reset successfully")
        return jsonify({
            "message": "All resources reset successfully",
            "status": "success"
        })
    except Exception as e:
        print(f"❌ Error during reset: {str(e)}")
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
