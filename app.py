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

# Import the user service handlers
from src.services.user_service import handle_set_name, handle_get_user_name
# Import the web crawl service handlers
from src.services.web_crawl_service import handle_crawl_and_index, process_crawl_request

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
    """Store user's name in session by calling the handler."""
    success, response_data, status_code = handle_set_name(request, session)
    return jsonify(response_data), status_code

@app.route('/get_user_name')
def get_user_name():
    """Get user's ID from session by calling the handler."""
    return handle_get_user_name(session)

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

@app.route('/crawl', methods=['POST'])
def crawl_website():
    """Handles website crawling requests by delegating to the service function."""
    global _document_metadata
    
    # Delegate request processing to the service function
    response_data, status_code, updated_metadata = process_crawl_request(
        request,
        session,
        get_index_name, # Pass the function itself
        app.logger,     # Pass the app's logger
        _document_metadata # Pass the current metadata
    )
    
    # Update the global metadata list with the result from the service
    _document_metadata = updated_metadata
    
    return jsonify(response_data), status_code

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
    
    # Log the response content before sending
    app.logger.info(f"Search successful. Sending response: {response[:500]}...") # Log first 500 chars
    
    return jsonify({
        "response": response,
        "user_context": user_id
    })

# @app.route('/delete_documents', methods=['POST'])
# def delete_documents():
#     """Delete specific documents from Pinecone and local metadata."""
#     global _document_metadata
    
#     try:
#         data = request.get_json()
#         document_ids = data.get('document_ids', [])
#         user_id = session.get('user_id', 'anonymous')
        
#         if not document_ids:
#             return jsonify({"error": "No document IDs provided"}), 400
            
#         index_name = get_index_name()
#         if not index_name:
#             return jsonify({"error": "No index available"}), 400
            
#         success, message = delete_user_data(index_name, user_id, document_ids)
        
#         if success:
#             # Update local document metadata
#             _document_metadata = [doc for doc in _document_metadata if doc.get('document_id') not in document_ids]
#             return jsonify({
#                 "message": message,
#                 "documents": _document_metadata
#             })
#         else:
#             return jsonify({"error": message}), 500
            
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the entire session, clearing all documents."""
    global _document_metadata
    
    try:
        user_id = session.get('user_id', 'anonymous')
        index_name = get_index_name()
        
        if index_name and _document_metadata: # Only delete if index and metadata exist
            # Collect all document IDs from the current metadata
            all_document_ids = [doc.get('document_id') for doc in _document_metadata if doc.get('document_id')]
            
            if all_document_ids: # If there are IDs to delete
                app.logger.info(f"Resetting session for {user_id}. Deleting {len(all_document_ids)} documents.")
                success, message = delete_user_data(index_name, user_id, all_document_ids)
                if not success:
                    app.logger.error(f"Failed to delete documents during reset for {user_id}: {message}")
                    return jsonify({"error": f"Failed to delete documents during reset: {message}"}), 500
            else:
                 app.logger.info(f"Resetting session for {user_id}. No documents found in metadata to delete.")

        # Clear local metadata regardless of deletion success (or if there was nothing to delete)
        _document_metadata = []
        app.logger.info(f"Local document metadata cleared for {user_id}.")
        
        # Clear uploads folder content - Added for a more complete reset
        user_upload_folder = os.path.join(UPLOAD_FOLDER, format_username(user_id))
        if os.path.exists(user_upload_folder):
            try:
                shutil.rmtree(user_upload_folder)
                app.logger.info(f"Cleared upload folder for {user_id}: {user_upload_folder}")
            except Exception as e:
                 app.logger.error(f"Error clearing upload folder {user_upload_folder} for {user_id}: {e}")
                 # Decide if this should be a blocking error or just a warning
                 # return jsonify({"error": f"Failed to clear user uploads: {e}"}), 500 

        # Optional: Clear session data if needed (be careful not to log out unintentionally)
        # session.clear() # This would require the user to log in again

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return jsonify({
            "message": "Session reset successfully. All documents and associated data have been cleared.",
            "status": "success"
        })
        
    except Exception as e:
        app.logger.error(f"Exception during session reset for {user_id}: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during session reset: {str(e)}"}), 500

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    
    run_simple('0.0.0.0', 8080, app, 
               use_reloader=True,
               use_debugger=True,
               threaded=True,
               passthrough_errors=True)
