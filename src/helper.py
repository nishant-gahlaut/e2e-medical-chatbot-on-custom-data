from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt, title_generation_prompt
from src.model_singletons import get_cached_llm, get_cached_embeddings
import re

def load_documents(file_path):
    """Loads a PDF document."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def process_file_bytes(file_bytes, filename):
    """Processes a file directly from its bytes using a temporary file.
    
    Args:
        file_bytes: The binary content of the file
        filename: Original filename (for metadata)
        
    Returns:
        Tuple of (documents, metadata) where metadata contains title and other info
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
    try:
        # Load documents from the temporary file
        print(f"Processing file: {filename}")
        documents = load_documents(temp_path)
        
        # Create initial chunks to extract title
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        initial_chunks = text_splitter.split_documents(documents)
        
        # Generate document title
        llm = load_llm()
        document_title = generate_document_title(initial_chunks[:2], llm)
        
        # Add metadata to documents
        for doc in documents:
            doc.metadata["source"] = filename
            doc.metadata["title"] = document_title
        
        # Create file metadata
        file_metadata = {
            "filename": filename,
            "title": document_title,
            "page_count": len(documents),
            "timestamp": int(os.path.getmtime(temp_path)) if os.path.exists(temp_path) else None
        }
            
        return documents, file_metadata
    finally:
        # Always clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def create_chunk(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings():
    """Generates embeddings for document chunks."""
    try:
        return get_cached_embeddings()
    except Exception as e:
        print(f"Error creating embeddings: {str(e)}")
        raise e

def create_pinecone_index(index_name):
    """Creates a Pinecone index if it does not exist."""
    load_dotenv()
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index_name = index_name
    
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return index_name

def create_pinecone_vector_store(index_name, embeddings, chunks):
    """Stores document embeddings in Pinecone and returns a retriever."""
    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}, search_type="similarity")
    return retriever

def load_llm():
    """Loads the Google Gemini AI model."""
    return get_cached_llm()

def create_rag_chain(llm, retriever, system_prompt, user_input):
    """Creates the RAG pipeline to process user queries."""
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

def generate_document_title(chunks, llm=None):
    """Generates a title for a document using an LLM based on the first few chunks.
    
    Args:
        chunks: List of document chunks, with the first few representing the document intro
        llm: The LLM to use (if None, will create a new instance)
        
    Returns:
        A generated title for the document
    """
    if not chunks or len(chunks) == 0:
        return "Untitled Document"
    
    # Initialize LLM if not provided
    if llm is None:
        llm = load_llm()
    
    try:
        # Take content from first 2 chunks (or fewer if not available)
        sample_size = min(2, len(chunks))
        intro_text = ""
        
        for i in range(sample_size):
            if hasattr(chunks[i], 'page_content'):
                intro_text += chunks[i].page_content + " "
            elif isinstance(chunks[i], str):
                intro_text += chunks[i] + " "
                
        # Limit text length (approximately 1000 tokens)
        intro_text = intro_text[:4000].strip()
        
        if not intro_text:
            return "Untitled Document"
        
        # Create title generation prompt
        prompt = ChatPromptTemplate.from_template(title_generation_prompt)
        
        # Get title from LLM
        title_response = llm.invoke(prompt.format(document_text=intro_text))
        
        # Extract response
        if hasattr(title_response, 'content'):
            title = title_response.content
        else:
            title = str(title_response)
        
        # Clean up title
        title = clean_title(title)
        print(f"Generated title: {title}")
        
        return title if title else "Untitled Document"
        
    except Exception as e:
        print(f"Error generating title: {str(e)}")
        return "Untitled Document"

def clean_title(title):
    """Cleans up a generated title by removing quotes, extra whitespace, etc."""
    if not title:
        return ""
    
    # Remove quotes and extra spaces
    title = re.sub(r'^["\'"\']|["\'"\']$', '', title.strip())
    title = re.sub(r'\s+', ' ', title)
    
    # Remove common prefixes LLMs might add
    prefixes_to_remove = ['title:', 'document title:', 'suggested title:']
    for prefix in prefixes_to_remove:
        if title.lower().startswith(prefix):
            title = title[len(prefix):].strip()
    
    # Ensure proper capitalization (if title is too lowercase)
    if not any(c.isupper() for c in title):
        title = title.title()
    
    # Limit length
    if len(title) > 50:
        title = title[:47] + "..."
        
    return title.strip()
