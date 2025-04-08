import os
import time
from flask import Flask, request, jsonify, render_template, url_for
from dotenv import load_dotenv
from src.service import get_response
import shutil
from src.helper import load_documents, create_chunk, create_embeddings, create_pinecone_index, create_pinecone_vector_store, load_llm, create_rag_chain
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
    """Handles multiple file uploads and processes them for vector storage."""
    global _documents, _chunks, _embeddings, _vector_store, _llm
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No selected files"}), 400
    
    try:
        # Save uploaded files
        file_paths = []
        for file in files:
            if file.filename:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                file_paths.append(file_path)
        
        # Process all uploaded files
        all_documents = []
        for file_path in file_paths:
            documents = load_documents(file_path)
            all_documents.extend(documents)
        
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
            "message": "Files uploaded and processed successfully", 
            "file_paths": file_paths,
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
    """Clears uploaded files and resets Pinecone index."""
    global _documents, _chunks, _embeddings, _vector_store, _llm
    
    try:
        # Reset global variables
        _documents = None
        _chunks = None
        _embeddings = None
        _vector_store = None
        _llm = None
        # Delete Pinecone index
        try:
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            for index in pc.list_indexes():
                pc.delete_index(index.name)
        except Exception as e:
            print(f"Error deleting Pinecone indexes: {str(e)}")
        
        # Delete uploaded files
        shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        return jsonify({"message": "Reset successful"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
