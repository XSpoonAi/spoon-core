#!/usr/bin/env python3
"""
You.com Search Script for Research Skill
This script uses the You.com search API to search for information on any topic.

Opt-in: it makes no request unless YOUCOM_SEARCH_ENABLED=1 is set.
It works without an API key (about 100 searches a day); set YDC_API_KEY for the keyed API.
"""

import json
import os
import sys

import requests

KEYLESS_SEARCH_URL = "https://api.you.com/v1/agents/search"
KEYED_SEARCH_URL = "https://ydc-index.io/v1/search"
USER_AGENT = "youdotcom-integration/XSpoonAi-spoon-core"
MAX_CONTENT_CHARS = 1000
KEYLESS_MAX_RESULTS = 50
ENABLE_ENV = "YOUCOM_SEARCH_ENABLED"


def is_enabled() -> bool:
    """You.com search runs only when the user sets YOUCOM_SEARCH_ENABLED=1."""
    return os.environ.get(ENABLE_ENV, "").strip().lower() in ("1", "true", "yes")


def search(query: str, max_results: int = 5) -> dict:
    """
    Search using the You.com API (keyless unless YDC_API_KEY is set).
    """
    if not is_enabled():
        return {
            "status": "error",
            "message": (
                "You.com search is disabled. It is opt-in: set "
                f"{ENABLE_ENV}=1 to send search queries to You.com."
            ),
        }

    headers = {"User-Agent": USER_AGENT}
    api_key = os.environ.get("YDC_API_KEY", "").strip()

    try:
        if api_key:
            headers["X-API-Key"] = api_key
            response = requests.post(
                KEYED_SEARCH_URL,
                json={"query": query, "count": max_results},
                headers=headers,
                timeout=20,
            )
        else:
            response = requests.get(
                KEYLESS_SEARCH_URL,
                params={"query": query, "count": min(max_results, KEYLESS_MAX_RESULTS)},
                headers=headers,
                timeout=20,
            )
        response.raise_for_status()

        # Format results; the page text is in "snippets", with "description" as a fallback
        results = []
        for item in (response.json().get("results") or {}).get("web") or []:
            text = " ".join(item.get("snippets") or []) or item.get("description") or ""
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": text[:MAX_CONTENT_CHARS],
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
