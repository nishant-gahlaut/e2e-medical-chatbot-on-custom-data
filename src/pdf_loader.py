from typing import List
from langchain_core.documents import Document
from pypdf import PdfReader

class PyPDFLoader:
    """Custom implementation of PDF loader without langchain-community dependency."""
    
    def __init__(self, file_path: str):
        """Initialize the PDF loader.
        
        Args:
            file_path: Path to the PDF file
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and parse the PDF file.
        
        Returns:
            List of Document objects, one per page with extracted text
        """
        reader = PdfReader(self.file_path)
        documents = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                metadata = {
                    "page": i + 1,
                    "source": self.file_path,
                    "total_pages": len(reader.pages)
                }
                documents.append(Document(
                    page_content=text.strip(),
                    metadata=metadata
                ))
        
        return documents 