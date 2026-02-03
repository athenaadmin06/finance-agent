import fitz  # PyMuPDF
import easyocr
import io
import numpy as np
from PIL import Image

# Initialize EasyOCR reader once (to avoid reloading model on every call)
reader = easyocr.Reader(['en'], gpu=False) 

def parse_document_local(file_obj):
    """
    Detects file type and routes to the appropriate parser.
    Args:
        file_obj (UploadedFile): Streamlit uploaded file object.
    Returns:
        str: Extracted text content.
    """
    filename = file_obj.name.lower()
    
    try:
        if filename.endswith('.pdf'):
            return _parse_pdf(file_obj)
        elif filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return _parse_image(file_obj)
        else:
            print(f"⚠️ Unsupported file format: {filename}")
            return ""
            
    except Exception as e:
        print(f"❌ Error parsing {filename}: {e}")
        return ""

def _parse_pdf(file_obj):
    """
    Extracts text from PDF.
    Strategy:
    1. Try extracting text directly (PyMuPDF).
    2. If text is empty (scanned PDF), fallback to OCR (EasyOCR) on page images.
    """
    file_obj.seek(0)
    file_bytes = file_obj.read()
    
    # Open PDF from bytes
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []
    
    for page_num, page in enumerate(doc):
        # A. Try direct text extraction
        text = page.get_text()
        
        # B. Fallback: If page has very little text, assume it's scanned -> Use OCR
        if len(text.strip()) < 10:  # Threshold for "empty" page
            # Render page as image (pixmap)
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            
            # Perform OCR on the image bytes
            text = _perform_ocr_on_bytes(img_bytes)
            
        full_text.append(text)
        
    return "\n".join(full_text)

def _parse_image(file_obj):
    """
    Extracts text from an image file using EasyOCR.
    """
    file_obj.seek(0)
    image_bytes = file_obj.read()
    return _perform_ocr_on_bytes(image_bytes)

def _perform_ocr_on_bytes(img_bytes):
    """
    Helper function to run EasyOCR on image bytes.
    """
    try:
        result = reader.readtext(img_bytes, detail=0) # detail=0 returns just the text list
        return " ".join(result)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""
