#!/bin/bash
# Run all experiments defined in run_experiments.py

# Ensure we are in the project root
cd "$(dirname "$0")/.."

echo "Starting Experiments..."
python3 experiment/run_experiments.py "$@"
echo "Experiments completed."
