#!/usr/bin/env bash
# =============================================================================
# Political Polarity Analysis - One-Command Run Script
# =============================================================================
# This script runs the full pipeline end-to-end on synthetic data.
# Usage: bash run_all.sh
# =============================================================================

set -euo pipefail

echo "=========================================="
echo " Political Polarity Analysis Pipeline"
echo "=========================================="
echo ""
echo "WARNING: This system classifies TEXT CONTENT, not individuals."
echo "Do NOT use outputs to infer political affiliation."
echo ""

# Step 1: Generate synthetic data
echo "--- Step 1: Generating synthetic data ---"
python -m scripts.generate_synthetic --n-per-class 50 --seed 42

# Step 2: Prepare data (preprocess, label, split)
echo "--- Step 2: Preparing data ---"
python -m scripts.prepare_data --config configs/default.yaml --skip-lang-filter

# Step 3: Train baseline model
echo "--- Step 3: Training baseline model ---"
python -m scripts.train_baseline --config configs/default.yaml

# Step 4: Evaluate baseline
echo "--- Step 4: Evaluating baseline model ---"
python -m scripts.evaluate --config configs/default.yaml --model-type baseline --split test

# Step 5: Generate subset reports
echo "--- Step 5: Generating subset reports ---"
python -m scripts.report_subsets --config configs/default.yaml --model-type baseline --group-by source region

# Step 6: Single-text inference demo
echo "--- Step 6: Inference demo ---"
python -m scripts.infer --config configs/default.yaml --model-type baseline \
    --text "The government should invest more in renewable energy and public healthcare programs." \
    --show-interpretability

echo ""
echo "=========================================="
echo " Pipeline complete!"
echo " Artifacts saved to: ./artifacts/"
echo "=========================================="
