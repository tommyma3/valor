"""
Google Scholar Search Tool for IterResearch.

Provides academic publication search functionality.
"""
import json
import time
import requests
from typing import List, Union, Optional
from concurrent.futures import ThreadPoolExecutor

try:
    from config import SCHOLAR_API_KEY, SCHOLAR_API_URL
except ImportError:
    import os
    SCHOLAR_API_KEY = os.getenv("SCHOLAR_API_KEY", "")
    SCHOLAR_API_URL = os.getenv("SCHOLAR_API_URL", "https://serpapi.com/search")


class Scholar:
    """
    Google Scholar search tool for academic publications.
    
    Supports:
    - Single query search
    - Batch queries with parallel execution
    - Returns publication metadata including citations
    """
    
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of search queries for Google Scholar."
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the scholar search tool.
        
        Args:
            api_key: API key for the search service.
            api_url: API URL for the search service.
        """
        self.api_key = api_key or SCHOLAR_API_KEY
        self.api_url = api_url or SCHOLAR_API_URL

    def _search_single(self, query: str) -> str:
        """
        Perform a single Google Scholar search.
        
        Args:
            query: The search query string.
            
        Returns:
            Formatted search results as a string.
        """
        params = {
            "api_key": self.api_key,
            "engine": "google_scholar",
            "q": query,
            "num": 10,
            "hl": "en",
        }
        
        max_retries = 5
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                results = response.json()
                
                organic_results = results.get("organic_results", [])
                
                if not organic_results:
                    return f"No results found for query: '{query}'. Use a less specific query."
                
                # Format results
                web_snippets = []
                for idx, page in enumerate(organic_results, 1):
                    title = page.get("title", "No title")
                    link = page.get("link", "no available link")
                    snippet = page.get("snippet", "")
                    
                    # Publication info
                    pub_info = page.get("publication_info", {})
                    summary = pub_info.get("summary", "")
                    
                    # Citation info
                    cited_by = page.get("inline_links", {}).get("cited_by", {})
                    citation_count = cited_by.get("total", "")
                    
                    # PDF link if available
                    resources = page.get("resources", [])
                    pdf_link = ""
                    for res in resources:
                        if res.get("file_format") == "PDF":
                            pdf_link = res.get("link", "")
                            break
                    
                    link_info = f"pdfUrl: {pdf_link}" if pdf_link else f"link: {link}"
                    pub_str = f"\npublicationInfo: {summary}" if summary else ""
                    cite_str = f"\ncitedBy: {citation_count}" if citation_count else ""
                    snippet_str = f"\n{snippet}" if snippet else ""
                    
                    formatted = f"{idx}. [{title}]({link_info}){pub_str}{cite_str}{snippet_str}"
                    web_snippets.append(formatted)
                
                content = f"A Google Scholar search for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n" + "\n\n".join(web_snippets)
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Google Scholar search failed for '{query}': {str(e)}"
                time.sleep(1)
        
        return f"No results found for '{query}'. Try with a more general query."

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute Google Scholar search with given parameters.
        
        Args:
            params: Either a JSON string or dict with 'query' field.
            
        Returns:
            Formatted search results.
        """
        if not self.api_key:
            return "[Google Scholar] Error: SCHOLAR_API_KEY is not configured. Please set it in environment variables."
        
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "[Google Scholar] Invalid request format: Input must be a JSON object containing 'query' field"
        
        query = params.get("query")
        if not query:
            return "[Google Scholar] Invalid request format: Input must contain 'query' field"
        
        # Handle single or multiple queries
        if isinstance(query, str):
            return self._search_single(query)
        elif isinstance(query, list):
            with ThreadPoolExecutor(max_workers=3) as executor:
                responses = list(executor.map(self._search_single, query))
            return "\n=======\n".join(responses)
        else:
            return "[Google Scholar] Invalid query format: must be string or array of strings"


if __name__ == "__main__":
    # Example usage
    tool = Scholar()
    print(tool.call({"query": ["transformer attention mechanism", "BERT language model"]}))
