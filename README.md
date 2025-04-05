# Custom Data Chatbot with RAG

A Flask-based chatbot that uses RAG (Retrieval Augmented Generation) to answer questions based on uploaded custom documents. The application uses Pinecone for vector storage, Google's Gemini for language processing, and HuggingFace embeddings for document processing.

## Features

- Upload and process custom documents (PDFs)
- Store document embeddings in Pinecone
- Query the documents using natural language
- Reset functionality to clear uploaded documents and Pinecone indexes

## Prerequisites

- Python 3.10 or higher
- Pinecone API key
- Google Gemini API key

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd e2e-medical-chatbot-on-custom-data
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_GEMINI_KEY=your_gemini_api_key
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:8080`

3. Upload your custom documents through the web interface

4. Ask questions about the uploaded documents

5. Use the reset button to clear all uploaded documents and Pinecone indexes

## Project Structure

- `app.py`: Main Flask application
- `src/`
  - `helper.py`: Utility functions for document processing and embeddings
  - `service.py`: Service layer for handling business logic
  - `prompt.py`: System prompts for the chatbot
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files
- `uploads/`: Directory for storing uploaded documents

## Dependencies

- Flask: Web framework
- LangChain: Framework for building LLM applications
- Pinecone: Vector database
- Google Gemini: Language model
- Sentence Transformers: For document embeddings
- PyPDF: For PDF processing

## License

[Add your license here]
