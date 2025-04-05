from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import system_prompt

def load_documents(file_path):
    """Loads a PDF document."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

def create_chunk(documents):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings():
    """Generates embeddings for document chunks."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings
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
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_GEMINI_KEY"),
        convert_system_message_to_human=True
    )
    return llm

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
