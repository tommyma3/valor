"""
PDF Parser Tool for IterResearch.

Provides PDF document parsing functionality using PyMuPDF.
"""
import os
import time
import uuid
import requests
from datetime import datetime
from typing import Optional

try:
    from config import SCRAPER_API_KEY
except ImportError:
    import os
    SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY", "")


def download_pdf(url: str, output_dir: str = "/tmp/pdf", timeout: int = 60) -> Optional[str]:
    """
    Download a PDF file from a URL.
    
    Args:
        url: URL of the PDF file.
        output_dir: Directory to save the downloaded file.
        timeout: Request timeout in seconds.
        
    Returns:
        Path to the downloaded file, or None if download failed.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"document_{current_time}_{str(uuid.uuid4())}.pdf"
    file_path = os.path.join(output_dir, filename)
    
    max_retries = 3
    
    # Method 1: Try direct download
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type or url.lower().endswith('.pdf'):
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Direct download failed: {str(e)}")
            time.sleep(1)
    
    # Method 2: Try ScraperAPI if configured
    if SCRAPER_API_KEY:
        scraper_params = {
            'api_key': SCRAPER_API_KEY,
            'url': url,
            'device_type': 'desktop',
            'country_code': 'us'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    'https://api.scraperapi.com/',
                    params=scraper_params,
                    timeout=timeout
                )
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"ScraperAPI download failed: {str(e)}")
                time.sleep(1)
    
    return None


def parse_pdf(pdf_path: str, use_cloud: bool = False) -> str:
    """
    Parse a PDF file and extract text content using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file.
        use_cloud: Deprecated parameter, kept for backward compatibility.
                   Always uses local PyMuPDF parsing.
        
    Returns:
        Extracted text content.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return "[PDF Parser Error]: PyMuPDF not installed. Run: pip install pymupdf"
    
    try:
        doc = fitz.open(pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"## Page {page_num + 1}\n\n{text}")
        
        doc.close()
        return '\n\n'.join(text_parts)
        
    except Exception as e:
        return f"[PDF Parser Error]: {str(e)}"


if __name__ == "__main__":
    # Test with a sample PDF
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Parsing: {pdf_path}")
        result = parse_pdf(pdf_path)
        print(result[:2000])  # Print first 2000 chars
    else:
        print("Usage: python pdf_parser.py <pdf_path>")
