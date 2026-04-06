import fitz  # PyMuPDF


def parse_document_local(file_obj):
    """
    Parses PDF documents and extracts text content (text-based PDFs only).
    """
    filename = file_obj.name.lower()
    
    try:
        if filename.endswith('.pdf'):
            return _parse_pdf(file_obj)
        else:
            print(f"⚠️ Unsupported file format: {filename}. Only PDF files are supported.")
            return ""
            
    except Exception as e:
        print(f"❌ Error parsing {filename}: {e}")
        return ""


def _parse_pdf(file_obj):
    """
    Extracts text from PDF using direct text extraction only.
    """
    file_obj.seek(0)
    file_bytes = file_obj.read()
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = []
    
    for page in doc:
        text = page.get_text()
        full_text.append(text)
    
    doc.close()
    return "\n".join(full_text)
