#!/usr/bin/env python3
"""
Tavily Search Script for Web3 Research Skill

This script uses Tavily API to search for Web3, cryptocurrency, and blockchain information.
The AI decides what query to pass via stdin.

Requirements:
- TAVILY_API_KEY environment variable must be set

Usage:
    echo "Ethereum Layer 2 solutions comparison" | python tavily_search.py
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

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        Search results as dictionary
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "message": "TAVILY_API_KEY environment variable not set"
        }

    try:
        client = TavilyClient(api_key=api_key)

        # Perform search with web3/crypto focus
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_domains=[
                "coindesk.com",
                "cointelegraph.com",
                "decrypt.co",
                "theblock.co",
                "defillama.com",
                "ethereum.org",
                "docs.chain.link",
                "messari.io",
                "dune.com"
            ]
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
