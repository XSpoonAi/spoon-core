#!/usr/bin/env python3
"""
Data transformation script for the data-processor skill.

Reads input from stdin, detects format, and can transform between formats.
The AI decides what data to pass and how to use the output.
"""

import sys
import json
import csv
import io
from typing import Any, Dict, List


def detect_format(text: str) -> str:
    """Detect the format of input text."""
    text = text.strip()

    # Try JSON
    try:
        json.loads(text)
        return "json"
    except json.JSONDecodeError:
        pass

    # Try CSV (check for comma-separated values with consistent columns)
    lines = text.split('\n')
    if len(lines) > 1:
        first_commas = lines[0].count(',')
        if first_commas > 0 and all(line.count(',') == first_commas for line in lines[:5] if line.strip()):
            return "csv"

    return "text"


def json_to_csv(data: Any) -> str:
    """Convert JSON to CSV format."""
    output = io.StringIO()

    if isinstance(data, list) and data and isinstance(data[0], dict):
        # List of dictionaries
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    elif isinstance(data, dict):
        # Single dictionary - convert to key,value pairs
        writer = csv.writer(output)
        writer.writerow(['key', 'value'])
        for key, value in data.items():
            writer.writerow([key, json.dumps(value) if isinstance(value, (dict, list)) else value])
    else:
        raise ValueError("Cannot convert this JSON structure to CSV")

    return output.getvalue()


def csv_to_json(text: str) -> List[Dict]:
    """Convert CSV to JSON format."""
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def text_to_json(text: str) -> Dict:
    """Convert plain text to JSON structure."""
    lines = text.strip().split('\n')
    return {
        "lines": lines,
        "line_count": len(lines),
        "content": text
    }


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

    # Detect input format
    input_format = detect_format(input_text)

    result = {
        "status": "success",
        "input_format": input_format,
        "transformations": {}
    }

    try:
        if input_format == "json":
            data = json.loads(input_text)
            result["transformations"]["to_csv"] = json_to_csv(data)
            result["transformations"]["formatted_json"] = json.dumps(data, indent=2)

        elif input_format == "csv":
            data = csv_to_json(input_text)
            result["transformations"]["to_json"] = json.dumps(data, indent=2)
            result["transformations"]["record_count"] = len(data)

        else:  # text
            data = text_to_json(input_text)
            result["transformations"]["to_json"] = json.dumps(data, indent=2)

    except Exception as e:
        result["status"] = "partial"
        result["error"] = str(e)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
