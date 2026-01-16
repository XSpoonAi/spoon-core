#!/usr/bin/env python3
"""
Data analysis script for the data-processor skill.

Reads input from stdin, analyzes it, and outputs statistics.
The AI decides what data to pass and how to interpret results.
"""

import sys
import json
from typing import Any, Dict


def analyze_value(value: Any, path: str = "root") -> Dict[str, Any]:
    """Analyze a value and return its characteristics."""
    result = {
        "path": path,
        "type": type(value).__name__,
    }

    if isinstance(value, dict):
        result["keys"] = list(value.keys())
        result["key_count"] = len(value)
    elif isinstance(value, list):
        result["length"] = len(value)
        if value:
            result["item_types"] = list(set(type(item).__name__ for item in value))
    elif isinstance(value, str):
        result["length"] = len(value)
        result["lines"] = value.count('\n') + 1
    elif isinstance(value, (int, float)):
        result["value"] = value

    return result


def analyze_structure(data: Any, max_depth: int = 3) -> Dict[str, Any]:
    """Analyze data structure recursively."""
    stats = {
        "root_type": type(data).__name__,
        "total_keys": 0,
        "total_items": 0,
        "max_depth": 0,
        "types_found": set(),
        "structure": []
    }

    def traverse(obj: Any, depth: int = 0, path: str = "root"):
        if depth > max_depth:
            return

        stats["max_depth"] = max(stats["max_depth"], depth)
        stats["types_found"].add(type(obj).__name__)

        if isinstance(obj, dict):
            stats["total_keys"] += len(obj)
            for key, value in obj.items():
                child_path = f"{path}.{key}"
                stats["structure"].append(analyze_value(value, child_path))
                traverse(value, depth + 1, child_path)

        elif isinstance(obj, list):
            stats["total_items"] += len(obj)
            for i, item in enumerate(obj[:5]):  # Only first 5 items
                child_path = f"{path}[{i}]"
                traverse(item, depth + 1, child_path)

    traverse(data)
    stats["types_found"] = list(stats["types_found"])

    return stats


def main():
    """Main entry point."""
    # Read input from stdin
    input_text = sys.stdin.read().strip()

    if not input_text:
        print(json.dumps({
            "status": "error",
            "message": "No input provided. Please pass data via stdin."
        }))
        return

    # Try to parse as JSON
    try:
        data = json.loads(input_text)
        result = {
            "status": "success",
            "format": "json",
            "analysis": analyze_structure(data)
        }
    except json.JSONDecodeError:
        # Treat as plain text
        lines = input_text.split('\n')
        result = {
            "status": "success",
            "format": "text",
            "analysis": {
                "total_lines": len(lines),
                "total_chars": len(input_text),
                "non_empty_lines": len([l for l in lines if l.strip()]),
                "avg_line_length": sum(len(l) for l in lines) / len(lines) if lines else 0
            }
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
