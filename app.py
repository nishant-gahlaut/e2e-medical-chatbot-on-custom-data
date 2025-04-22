import os
import time
import threading
from flask import Flask, request, jsonify, render_template, url_for, session, redirect
from dotenv import load_dotenv
from src.service import (
    process_uploaded_files,
    delete_user_data,
    process_search_query
)
from src.helper import (
    get_or_create_index,
    format_username
)
import shutil
from src.helper import (
    load_documents, process_file_bytes, create_chunk, create_embeddings,
    create_pinecone_vector_store, create_rag_chain,
    log_memory_usage
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
    
    def initialize_pinecone():
        """Initialize Pinecone index and return the index name."""
        try:
            index_name = get_or_create_index()
            app.logger.info(f"Initialized Pinecone index: {index_name}")
            return index_name
        except Exception as e:
            app.logger.error(f"Error initializing Pinecone: {str(e)}")
            return None
    
    # Initialize Pinecone index
    index_name = initialize_pinecone()
    if index_name:
        app.config['INDEX_NAME'] = index_name
    
    return app

# Create the Flask application
app = create_app()

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_GEMINI_KEY = os.getenv("GOOGLE_GEMINI_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_GEMINI_KEY"] = GOOGLE_GEMINI_KEY

# Global state for document metadata only
_document_metadata = []

def get_index_name():
    """Get the current index name from app config."""
    return app.config.get('INDEX_NAME')

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
    user_id = session.get('user_id', 'anonymous')
    index_name = get_index_name()
    
    response, success, error_message = process_search_query(query, user_id, index_name)
    
    if not success:
        return jsonify({"error": error_message}), 400 if "required" in error_message else 500
    
    return jsonify({
        "response": response,
        "user_context": user_id
    })

@app.route('/delete_documents', methods=['POST'])
def delete_documents():
    """Delete specific documents from Pinecone and local metadata."""
    global _document_metadata
    
    try:
        data = request.get_json()
        document_ids = data.get('document_ids', [])
        user_id = session.get('user_id', 'anonymous')
        
        if not document_ids:
            return jsonify({"error": "No document IDs provided"}), 400
            
        index_name = get_index_name()
        if not index_name:
            return jsonify({"error": "No index available"}), 400
            
        success, message = delete_user_data(index_name, user_id, document_ids)
        
        if success:
            # Update local document metadata
            _document_metadata = [doc for doc in _document_metadata if doc.get('document_id') not in document_ids]
            return jsonify({
                "message": message,
                "documents": _document_metadata
            })
        else:
            return jsonify({"error": message}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the entire session, clearing all documents."""
    global _document_metadata
    
    try:
        user_id = session.get('user_id', 'anonymous')
        index_name = get_index_name()
        
        if index_name:
            success, message = delete_user_data(index_name, user_id)
            if not success:
                return jsonify({"error": message}), 500
        
        _document_metadata = []
        
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
