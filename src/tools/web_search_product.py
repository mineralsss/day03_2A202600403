import os
import requests

def web_search_product(query: str) -> str:
    """Search for a product on the web using SerpAPI."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key or api_key == "your_serpapi_api_key_here":
        return "Error: Bạn cần cấu hình SERPAPI_API_KEY trong file .env để tìm kiếm online."

    params = {
        "engine": "google",
        "q": f"giá cả tính năng {query}",
        "api_key": api_key,
        "hl": "vi",  # Vietnamese language
        "gl": "vn"   # Location Vietnam
    }

    try:
        response = requests.get("https://serpapi.com/search", params=params)
        response.raise_for_status()
        data = response.json()

        # Extract organic results snippets
        if "organic_results" not in data or not data["organic_results"]:
            return f"Không tìm thấy thông tin online cho: {query}"

        results = []
        for i, item in enumerate(data["organic_results"][:3]):  # Top 3 results
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            results.append(f"{i+1}. {title}\nURL: {link}\nSnippet: {snippet}")

        return (
            f"Kết quả tìm kiếm web cho '{query}':\n" + 
            "\n".join(results)
        )

    except Exception as e:
        return f"Lỗi khi tìm kiếm online: {str(e)}"

TOOL_SPEC = {
    "name": "web_search_product",
    "description": "Tìm kiếm thông tin, tính năng và giá cả của một sản phẩm trên mạng Internet nếu sản phẩm không có trong kho (database). Input: từ khóa tìm kiếm, ví dụ 'iPhone 15 Pro Max'.",
    "func": web_search_product
}
