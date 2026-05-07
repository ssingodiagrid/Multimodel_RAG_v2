import fitz  # PyMuPDF
import pdfplumber
import base64
from PIL import Image
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from typing import List
from configs.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Gemini for summarization
        from langchain_google_vertexai import ChatVertexAI
        self.llm = ChatVertexAI(
            model_name=settings.gemini_model,
            project=settings.gcp_project,
            location=settings.gcp_location,
            temperature=0.0
        )

    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from the PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                full_text += page.get_text()
            return full_text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise

    def _summarize_table(self, table_text: str) -> str:
        prompt = f"Summarize the following tabular financial data in a detailed text paragraph. Include key numbers and trends. Table:\n\n{table_text}"
        return self.llm.invoke(prompt).content

    def _caption_image(self, image_bytes: bytes) -> str:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this chart or graph in detail, extracting all key data points, trends, labels, and axes. Be extremely precise with numbers."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        )
        return self.llm.invoke([message]).content

    def get_document_chunks(self, pdf_path: str) -> List[Document]:
        """Extract text, tables, and images page-by-page, chunk it, and attach metadata."""
        logger.info(f"Extracting text, tables, and images from {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            plumber_doc = pdfplumber.open(pdf_path)
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
            
        all_chunks = []
        chunk_id = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            plumber_page = plumber_doc.pages[page_num]
            
            # 1. TEXT EXTRACTION
            text = page.get_text()
            if text.strip():
                page_chunks = self.text_splitter.create_documents([text])
                for chunk in page_chunks:
                    chunk.metadata = {
                        "source": pdf_path,
                        "chunk_id": chunk_id,
                        "page_number": page_num + 1,
                        "content_type": "text"
                    }
                    all_chunks.append(chunk)
                    chunk_id += 1
                    
            # 2. TABLE EXTRACTION
            tables = plumber_page.extract_tables()
            for table in tables:
                if not table: continue
                # Flatten table into string
                table_text = "\n".join(["\t".join([str(cell) if cell else "" for cell in row]) for row in table])
                if len(table_text.strip()) > 20:
                    summary = self._summarize_table(table_text)
                    doc_table = Document(
                        page_content=f"TABLE SUMMARY:\n{summary}\n\nRAW TABLE:\n{table_text}",
                        metadata={
                            "source": pdf_path,
                            "chunk_id": chunk_id,
                            "page_number": page_num + 1,
                            "content_type": "table"
                        }
                    )
                    all_chunks.append(doc_table)
                    chunk_id += 1

            # 3. IMAGE EXTRACTION
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Check if it's a reasonably sized image (avoid tiny icons)
                img_obj = Image.open(io.BytesIO(image_bytes))
                if img_obj.width > 200 and img_obj.height > 200:
                    try:
                        caption = self._caption_image(image_bytes)
                        doc_image = Document(
                            page_content=f"IMAGE/CHART CAPTION:\n{caption}",
                            metadata={
                                "source": pdf_path,
                                "chunk_id": chunk_id,
                                "page_number": page_num + 1,
                                "content_type": "image"
                            }
                        )
                        all_chunks.append(doc_image)
                        chunk_id += 1
                    except Exception as e:
                        logger.warning(f"Failed to caption image on page {page_num + 1}: {e}")

        logger.info(f"Created {len(all_chunks)} chunks with multimodal metadata.")
        return all_chunks
