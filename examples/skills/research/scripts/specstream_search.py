#!/usr/bin/env python3
"""
SpecStream Search Script for Research Skill
This script uses SpecStream API to search for information on any topic.
"""

import os
import sys
import json
import requests
from typing import Optional

def search(query: str, max_results: int = 5) -> dict:
    """
    Search using SpecStream API.
    """
    api_key = os.environ.get("SPECSTREAM_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "message": "SPECSTREAM_API_KEY environment variable not set"
        }

    try:
        # SpecStream API Endpoint
        url = "https://api.specstream.ai/v1/web-search"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "query": query,
            "max_results": max_results
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        # Format results
        results = []
        for item in data.get("results", []):
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
            "answer": data.get("answer", "")
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def main():
    """Main entry point."""
    query = sys.stdin.read().strip()
    if not query:
        print(json.dumps({
            "status": "error",
            "message": "No query provided. Pass query via stdin."
        }))
        return

    result = search(query)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
