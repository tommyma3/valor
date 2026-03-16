"""
Google Search Tool for IterResearch.

Provides web search functionality through configurable search APIs.
Supports both single and batch queries.
"""
import json
import time
import requests
from typing import List, Union, Optional

try:
    from config import SEARCH_API_KEY, SEARCH_API_URL
except ImportError:
    import os
    SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
    SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://serpapi.com/search")


class Search:
    """
    Web search tool that performs Google searches.
    
    Supports:
    - Single query search
    - Batch queries (multiple queries in one call)
    - Configurable search API backend
    """
    
    name = "google_search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the search tool.
        
        Args:
            api_key: API key for the search service. Falls back to config if not provided.
            api_url: API URL for the search service. Falls back to config if not provided.
        """
        self.api_key = api_key or SEARCH_API_KEY
        self.api_url = api_url or SEARCH_API_URL
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
    
    def get_stats(self) -> dict:
        """Get search statistics."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": f"{(self.successful_requests / self.total_requests * 100):.2f}%" if self.total_requests > 0 else "0%"
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_requests = 0
        self.successful_requests = 0
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)

    def _search_single(self, query: str) -> str:
        """
        Perform a single search query.
        
        Args:
            query: The search query string.
            
        Returns:
            Formatted search results as a string.
        """
        self.total_requests += 1
        
        # Determine language settings based on query
        if self._contains_chinese(query):
            params = {
                "api_key": self.api_key,
                "q": query,
                "num": 10,
                "hl": "zh-CN",
                "gl": "CN",
            }
        else:
            params = {
                "api_key": self.api_key,
                "q": query,
                "num": 10,
                "hl": "en",
                "gl": "US",
            }
        
        max_retries = 5
        empty_result_retries = 0
        max_empty_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                results = response.json()
                
                # Extract organic results
                organic_results = results.get("organic_results", [])
                
                if len(organic_results) == 0:
                    empty_result_retries += 1
                    if empty_result_retries >= max_empty_retries:
                        return f"No results found for '{query}'. Try with a more general query."
                    time.sleep(1)
                    continue
                
                # Format results
                web_snippets = []
                for idx, page in enumerate(organic_results, 1):
                    title = page.get("title", "No title")
                    link = page.get("link", "")
                    snippet = page.get("snippet", "")
                    date = page.get("date", "")
                    source = page.get("source", "")
                    
                    date_str = f"\nDate published: {date}" if date else ""
                    source_str = f"\nSource: {source}" if source else ""
                    snippet_str = f"\n{snippet}" if snippet else ""
                    
                    formatted = f"{idx}. [{title}]({link}){date_str}{source_str}{snippet_str}"
                    web_snippets.append(formatted)
                
                self.successful_requests += 1
                content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Search failed for '{query}': {str(e)}"
                time.sleep(2)
        
        return f"No results found for '{query}'. Try with a more general query."
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """
        Execute search with given parameters.
        
        Args:
            params: Either a JSON string or dict with 'query' field.
                   Query can be a single string or list of strings.
                   
        Returns:
            Formatted search results.
        """
        if not self.api_key:
            return "[Search] Error: SEARCH_API_KEY is not configured. Please set it in environment variables."
        
        # Parse parameters
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        query = params.get("query")
        if not query:
            return "[Search] Invalid request format: Input must contain 'query' field"
        
        # Handle single or multiple queries
        if isinstance(query, str):
            return self._search_single(query)
        elif isinstance(query, list):
            responses = [self._search_single(q) for q in query]
            return "\n=======\n".join(responses)
        else:
            return "[Search] Invalid query format: must be string or array of strings"


if __name__ == "__main__":
    # Example usage
    search_tool = Search()
    
    print("--- Testing with a single query ---")
    params = {"query": ["Python programming tutorial"]}
    result = search_tool.call(params)
    print(result)
    print(f"\nStats: {search_tool.get_stats()}")
