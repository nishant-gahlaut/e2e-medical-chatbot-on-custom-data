import os
import time
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv
from src.service import get_response
import shutil
from src.helper import load_documents, process_file_bytes, create_chunk, create_embeddings, create_pinecone_index, create_pinecone_vector_store, load_llm, create_rag_chain
from src.prompt import system_prompt
from pinecone import Pinecone

load_dotenv()

app = Flask(__name__, 
            static_url_path='', 
            static_folder='static',
            template_folder='templates')
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
_llm = None
_index_name = None

@app.route('/')
def index():
    """Render chat UI."""
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handles multiple file uploads and processes them for vector storage without permanent storage."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _index_name
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    
    try:
        # Process files directly without saving to permanent storage
        all_documents = []
        processed_files = []
        
        # Maximum file size (10MB)
        MAX_FILE_SIZE = 10 * 1024 * 1024
        
        for file in files:
            if file.filename:
                # Check file size
                file.seek(0, os.SEEK_END)
                file_size = file.tell()
                file.seek(0)
                
                if file_size > MAX_FILE_SIZE:
                    return jsonify({"error": f"File {file.filename} exceeds the 10MB size limit"}), 400
                
                # Read file bytes
                file_bytes = file.read()
                
                # Process the file bytes directly using a temporary file
                documents = process_file_bytes(file_bytes, file.filename)
                all_documents.extend(documents)
                processed_files.append(file.filename)
        
        # Create chunks and embeddings
        _documents = all_documents
        _chunks = create_chunk(_documents)
        
        # Process in smaller batches if many chunks
        BATCH_SIZE = 50
        all_batches = [_chunks[i:i + BATCH_SIZE] for i in range(0, len(_chunks), BATCH_SIZE)]
        
        _embeddings = create_embeddings()
        _index_name = f"medical-docs-{int(time.time())}"
        
        # Create or get Pinecone index
        try:
            _index_name = create_pinecone_index(_index_name)
        except Exception as e:
            # Index might already exist
            print(f"Index creation error (might already exist): {str(e)}")
        
        # Create vector store processing batches to avoid memory issues
        for i, batch in enumerate(all_batches):
            if i == 0:
                # First batch creates the store
                _vector_store = create_pinecone_vector_store(_index_name, _embeddings, batch)
            else:
                # Add subsequent batches
                PineconeVectorStore.from_documents(
                    documents=batch,
                    embedding=_embeddings,
                    index_name=_index_name,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY")
                )
        
        return jsonify({
            "message": "Files processed successfully", 
            "processed_files": processed_files,
            "document_count": len(_documents),
            "chunk_count": len(_chunks)
        })
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
            _llm = load_llm()
        
        # Use the stored components to get a response
        response = create_rag_chain(_llm, _vector_store, system_prompt, query)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in chatbot endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Deletes Pinecone indexes and resets server-side resources."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _index_name
    
    try:
        # Delete Pinecone index first (to free up external resources)
        if _index_name:
            try:
                pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
                # Only delete our specific index
                pc.delete_index(_index_name)
            except Exception as e:
                print(f"Error deleting Pinecone index: {str(e)}")
        
        # Reset global variables to free memory
        _documents = None
        _chunks = None
        _embeddings = None
        _vector_store = None
        _llm = None
        _index_name = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return jsonify({"message": "Files deleted successfully"})
    except Exception as e:
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
