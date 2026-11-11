#!/usr/bin/env python3
"""
Keenable Search Script for Research Skill
This script uses the Keenable search API to search for information on any topic.
It works without an API key; set KEENABLE_API_KEY for higher rate limits.
"""

import json
import os
import sys

import requests

PUBLIC_SEARCH_URL = "https://api.keenable.ai/v1/search/public"
KEYED_SEARCH_URL = "https://api.keenable.ai/v1/search"
APP_TITLE = "spoon-core"
MAX_CONTENT_CHARS = 1000


def search(query: str, max_results: int = 5) -> dict:
    """
    Search using the Keenable API (keyless unless KEENABLE_API_KEY is set).
    """
    headers = {"X-Keenable-Title": APP_TITLE}
    api_key = os.environ.get("KEENABLE_API_KEY", "").strip()
    if api_key:
        url = KEYED_SEARCH_URL
        headers["X-API-Key"] = api_key
    else:
        url = PUBLIC_SEARCH_URL

    try:
        response = requests.post(
            url,
            json={"query": query, "max_results": max_results},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()

        # Format results; the page text is in "snippet"
        results = []
        for item in response.json().get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": (item.get("snippet") or "")[:MAX_CONTENT_CHARS],
            })

        return {
            "status": "success",
            "query": query,
            "results": results,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


def read_query(raw: str) -> str:
    """Accept a plain query or a JSON object with "query" or "input"."""
    raw = raw.strip()
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(data, dict):
        return str(data.get("query") or data.get("input") or "").strip()
    return raw


def main():
    """Main entry point."""
    # Read query from stdin
    query = read_query(sys.stdin.read())

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
