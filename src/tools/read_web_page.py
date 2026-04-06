import requests
from bs4 import BeautifulSoup
import re

def read_web_page(url: str) -> str:
    """Read the text content of a web page."""
    try:
        # Add basic headers to prevent getting blocked by some sites
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Adding a timeout so the agent doesn't hang forever
        response = requests.get(url.strip(), headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit text length to prevent overflowing the local model context window
        if len(text) > 4000:
            text = text[:4000] + "\n... (nội dung quá dài đã được cắt bớt)"
            
        return f"Nội dung từ {url}:\n{text}"
        
    except requests.exceptions.Timeout:
        return f"Lỗi: Quá thời gian tải trang {url}"
    except Exception as e:
        return f"Lỗi khi đọc trang web {url}: {str(e)}"

TOOL_SPEC = {
    "name": "read_web_page",
    "description": "True nếu cần đọc chi tiết nội dung của một bài viết hoặc trang web để lấy thông tin. Input: một đường link (URL) hợp lệ (ví dụ: https://example.com/article).",
    "func": read_web_page
}
