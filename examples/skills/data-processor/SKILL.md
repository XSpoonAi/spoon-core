---
name: data-processor
description: Data processing skill with Python and shell scripts for file analysis and transformation
version: 1.0.0
author: XSpoonAi Team
tags:
  - data
  - processing
  - analysis
  - scripts
triggers:
  - type: keyword
    keywords:
      - process
      - analyze
      - transform
      - data
      - parse
      - convert
    priority: 80
  - type: pattern
    patterns:
      - "(?i)(process|analyze|transform) .*(data|file|json|csv)"
      - "(?i)convert .* to .*"
    priority: 75
parameters:
  - name: input
    type: string
    required: false
    description: Input data or file path to process
  - name: format
    type: string
    required: false
    default: json
    description: Output format (json, csv, text)
composable: true
persist_state: false

scripts:
  enabled: true
  working_directory: ./scripts
  definitions:
    - name: analyze
      description: Analyze input data and provide statistics
      type: python
      file: analyze.py
      timeout: 30

    - name: transform
      description: Transform data format
      type: python
      file: transform.py
      timeout: 30

    - name: setup
      description: Initialize processing environment
      type: bash
      inline: |
        echo "Initializing data processor environment..."
        echo "Ready for processing"
      run_on_activation: true

    - name: cleanup
      description: Clean up temporary files
      type: bash
      inline: |
        echo "Cleaning up temporary files..."
        echo "Cleanup complete"
      run_on_deactivation: true
---

# Data Processor Skill

You are now operating in **Data Processing Mode**. You have access to scripts that can help process and analyze data.

## Available Scripts

### analyze
Analyzes input data and provides statistics. Pass data via stdin.

**Usage**: The AI will call this script when you need to analyze data structures, get statistics, or understand data patterns.

### transform
Transforms data between formats. Supports JSON, CSV, and text.

**Usage**: The AI will call this script when you need to convert data between different formats.

### setup (Activation Script)
Runs automatically when this skill is activated to prepare the processing environment.

### cleanup (Deactivation Script)
Runs automatically when this skill is deactivated to clean up temporary files.

## Guidelines

1. **Always validate input** before processing
2. **Handle errors gracefully** and provide informative messages
3. **Preserve data integrity** during transformations
4. **Report statistics** when analyzing data

## Example Tasks

1. "Analyze this JSON data and tell me about its structure"
2. "Convert this CSV to JSON format"
3. "Process this log file and extract key metrics"
