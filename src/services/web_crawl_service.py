# Placeholder for web crawl service logic 

import os
from firecrawl import FirecrawlApp
from langchain_core.documents import Document
from src.helper import create_chunk, create_embeddings, create_pinecone_vector_store
from src.model_singletons import get_cached_embeddings
import logging
import gc
import torch
from flask import jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handle_crawl_and_index(url: str, user_id: str, index_name: str):
    """Scrapes a website using Firecrawl, processes the content, and stores it in Pinecone."""
    
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        logger.error("Firecrawl API key not found in environment variables.")
        return False, "Server configuration error: Firecrawl API key missing.", None

    try:
        # Initialize FirecrawlApp
        app = FirecrawlApp(api_key=firecrawl_api_key)

        # Scrape the URL, passing formats directly as a keyword argument
        logger.info(f"Starting crawl for URL: {url}")
        scraped_data = app.scrape_url(url, formats=['markdown'])
        
        # Log the raw response from Firecrawl for debugging
        logger.info(f"Raw scraped data received from Firecrawl: {scraped_data}")

        # Check if scrape was successful and markdown content exists using attribute access
        if not scraped_data or not getattr(scraped_data, 'success', False) or not getattr(scraped_data, 'markdown', None):
             logger.error(f"Failed to scrape or extract markdown from URL: {url}. Success status: {getattr(scraped_data, 'success', 'N/A')}")
             return False, f"Could not scrape content from the provided URL: {url}", None

        # Access content and metadata using attribute notation
        content = scraped_data.markdown
        metadata = scraped_data.metadata if getattr(scraped_data, 'metadata', None) else {}
        page_title = metadata.get('title', 'No Title Found')
        source_url = metadata.get('sourceURL', url)

        logger.info(f"Successfully scraped content from: {source_url} (Title: {page_title})")

        # Create a Langchain Document
        doc = Document(
            page_content=content,
            metadata={
                "source": source_url,
                "title": page_title,
                "user_id": user_id,
                "document_type": "web_crawl"
                # Add any other relevant metadata from Firecrawl if needed
            }
        )

        # Process the document (chunking, embedding, storing)
        # Reuse existing helper functions
        embeddings = get_cached_embeddings()
        if not embeddings:
             logger.error("Failed to get cached embeddings model.")
             return False, "Server configuration error: Embeddings model not available.", None

        chunks = create_chunk([doc]) # create_chunk expects a list
        logger.info(f"Created {len(chunks)} chunks from the crawled content.")

        vector_store = create_pinecone_vector_store(index_name, embeddings, chunks)
        
        if not vector_store:
            logger.error("Failed to create or update Pinecone vector store.")
            return False, "Failed to store crawled content.", None

        logger.info(f"Successfully indexed content from {source_url} for user {user_id} in index {index_name}.")
        
        # Prepare document metadata for the response
        document_metadata = {
            "document_id": f"web_{user_id}_{source_url}", # Create a unique ID
            "filename": page_title, # Use title as filename representation
            "source_url": source_url,
            "document_type": "web_crawl",
            "chunk_count": len(chunks)
        }
        
        return True, f"Successfully crawled and indexed content from {source_url}.", document_metadata

    except Exception as e:
        # Log the full exception traceback for detailed debugging
        logger.error(f"Error during web crawl and indexing for URL {url}", exc_info=True) 
        # Try to extract specific error message if available
        error_message = str(e)
        # Potentially add more specific Firecrawl error handling if the library raises custom exceptions
        return False, f"An error occurred while processing the URL: {error_message}", None

def process_crawl_request(request, session, get_index_name, app_logger, current_metadata):
    """Handles the logic for a crawl request, including validation, execution, and response preparation."""
    
    user_id = session.get('user_id')
    if not user_id:
        return {"error": "User session not found. Please set a name first."}, 401, current_metadata

    data = request.get_json()
    url = data.get('url')
    if not url:
        return {"error": "URL is required"}, 400, current_metadata

    index_name = get_index_name()
    if not index_name:
        app_logger.error("Index name not configured.")
        return {"error": "Server configuration error: Index not available."}, 500, current_metadata

    app_logger.info(f"Received crawl request for URL: {url} from user: {user_id}")
    success, message, document_meta = handle_crawl_and_index(url, user_id, index_name)
    
    updated_metadata = current_metadata
    if success:
        if not updated_metadata:
             updated_metadata = []
        updated_metadata.append(document_meta)
        app_logger.info(f"Successfully processed and added metadata for crawled URL: {url}")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        response_data = {
            "message": message,
            "documents": updated_metadata,
            "document_count": len(updated_metadata)
        }
        return response_data, 200, updated_metadata
    else:
        app_logger.error(f"Failed to process crawl request for URL {url}: {message}")
        status_code = 500
        if "configuration error" in message or "API key" in message:
            status_code = 500
        elif "Could not scrape" in message:
            status_code = 400
        
        return {"error": message}, status_code, current_metadata # Return original metadata on failure 