import os
import time
import threading
from flask import Flask, request, jsonify, render_template, url_for, session, redirect
from dotenv import load_dotenv
from src.service import get_response, process_uploaded_files, delete_user_data
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

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__,
                static_url_path='',
                static_folder='static',
                template_folder='templates')
    
    app.secret_key = os.urandom(24)
    app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
    
    # Initialize Pinecone and models
    try:
        if not app.config.get('INDEX_NAME'):
            index_name = get_or_create_index()
            app.config['INDEX_NAME'] = index_name
            app.logger.info(f"Created new Pinecone index: {index_name}")
        
        # Initialize models
        _llm = get_cached_llm()
        _embeddings = get_cached_embeddings()
        
    except Exception as e:
        app.logger.error(f"Error initializing app: {str(e)}")
    
    return app

# Create the Flask application
app = create_app()

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_GEMINI_KEY"] = GOOGLE_GEMINI_KEY

_documents = None
_chunks = None
_embeddings = None
_vector_store = None
_llm = None
_document_metadata = []

def get_index_name():
    """Get the current index name from app config."""
    return app.config.get('INDEX_NAME')

def set_index_name(name):
    """Set the index name in app config."""
    app.config['INDEX_NAME'] = name

@app.route('/')
def index():
    """Render landing page."""
    if 'user_id' in session:
        return redirect(url_for('chat'))
    return render_template('landing.html')

@app.route('/chat')
def chat():
    """Render chat UI."""
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
    
    user_id = format_username(name)
    if not user_id:
        return jsonify({"success": False, "error": "Invalid name format"}), 400
    
    session['user_id'] = user_id
    
    return jsonify({"success": True, "redirect": url_for('chat')})

@app.route('/get_user_name')
def get_user_name():
    """Get user's ID from session."""
    return jsonify({"name": session.get('user_id', '')})

@app.route('/upload', methods=['POST'])
def upload():
    """Handles multiple file uploads and processes them for vector storage."""
    global _document_metadata
    
    user_id = session.get('user_id', 'anonymous')
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    
    response_data, success, error_message = process_uploaded_files(files, user_id)
    
    if not success:
        return jsonify({"error": error_message}), 500
        
    # Update document metadata
    if not _document_metadata:
        _document_metadata = []
    _document_metadata.extend(response_data["documents"])
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Return all documents
    return jsonify({
        "message": response_data["message"],
        "documents": _document_metadata,
        "document_count": len(_document_metadata),
        "chunk_count": response_data["chunk_count"],
        "index_name": response_data["index_name"]
    })

@app.route('/documents', methods=['GET'])
def get_documents():
    """Returns the list of uploaded documents with metadata."""
    global _document_metadata
    
    if not _document_metadata:
        return jsonify({"documents": [], "message": "No documents uploaded yet"})
    
    return jsonify({"documents": _document_metadata})

@app.route("/search", methods=["GET"])
def chatbot():
    """Handles user queries by fetching relevant information."""
    query = request.args.get("query")
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        # Get the current user_id and index_name
        user_id = session.get('user_id', 'anonymous')
        index_name = get_index_name()
        
        if not index_name:
            return jsonify({"error": "No index available"}), 400
            
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
        
        return jsonify({
            "response": response,
            "user_context": user_id
        })
    except Exception as e:
        app.logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Clears uploaded documents from the library."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _document_metadata
    
    try:
        _documents = None
        _chunks = None
        _embeddings = None
        _vector_store = None
        _document_metadata = []
        
        gc.collect()
        
        return jsonify({
            "message": "Document library cleared successfully",
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Delete user's data from Pinecone index and reset the session."""
    try:
        # Get user ID and index name
        user_id = session.get('user_id', 'anonymous')
        index_name = get_index_name()
        
        if index_name:
            # Delete only the user's data from the index
            success, message = delete_user_data(index_name, user_id)
            if not success:
                return jsonify({"error": f"Error deleting user data: {message}"}), 500
        
        # Reset application state
        reset()
        
        return jsonify({
            "message": "Session reset successfully",
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    
    run_simple('0.0.0.0', 8080, app, 
               use_reloader=True,
               use_debugger=True,
               threaded=True,
               passthrough_errors=True)
