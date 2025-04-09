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
        
        for file in files:
            if file.filename:
                # Read file bytes
                file_bytes = file.read()
                
                # Process the file bytes directly using a temporary file
                documents = process_file_bytes(file_bytes, file.filename)
                all_documents.extend(documents)
                processed_files.append(file.filename)
        
        # Create chunks and embeddings
        _documents = all_documents
        _chunks = create_chunk(_documents)
        _embeddings = create_embeddings()
        _index_name = f"medical-docs-{int(time.time())}"
        
        # Create or get Pinecone index
        try:
            _index_name = create_pinecone_index(_index_name)
        except Exception as e:
            # Index might already exist
            print(f"Index creation error (might already exist): {str(e)}")
        
        # Create vector store
        _vector_store = create_pinecone_vector_store(_index_name, _embeddings, _chunks)
        
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
    """Clears chat context and resets Pinecone index."""
    global _documents, _chunks, _embeddings, _vector_store, _llm, _index_name
    
    try:
        # Reset global variables
        _documents = None
        _chunks = None
        _embeddings = None
        _vector_store = None
        _llm = None
        _index_name = None
        
        # Delete Pinecone index
        try:
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            for index in pc.list_indexes():
                pc.delete_index(index.name)
        except Exception as e:
            print(f"Error deleting Pinecone indexes: {str(e)}")
        
        # No longer cleaning up file system as we're not storing files permanently
        
        return jsonify({"message": "Chat reset successful"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
