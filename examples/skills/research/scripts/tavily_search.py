#!/usr/bin/env python3
"""
General Search Script for Research Skill
This script uses Tavily API to search for information on any topic.
"""

import os
import sys
import json
from typing import Optional

try:
    from tavily import TavilyClient
except ImportError:
    print(json.dumps({
        "status": "error",
        "message": "Tavily package not installed. Run: pip install tavily-python"
    }))
    sys.exit(1)


def search(query: str, max_results: int = 5) -> dict:
    """
    Search using Tavily API.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "message": "TAVILY_API_KEY environment variable not set"
        }

    try:
        client = TavilyClient(api_key=api_key)

        # General search without domain restrictions
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results
        )

        # Format results
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0)
            })

        return {
            "status": "success",
            "query": query,
            "results": results,
            "answer": response.get("answer", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def main():
    """Main entry point."""
    # Read query from stdin
    query = sys.stdin.read().strip()

    if not query:
        print(json.dumps({
            "status": "error",
            "message": "No query provided. Pass query via stdin."
        }))
        return

    # Perform search
    result = search(query)

    # Output JSON result
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

