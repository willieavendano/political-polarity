# Political Polarity Text Analyzer

> **Teaching exemplar.** Built by the author, working with AI co-authoring agents, as an exemplar for a secondary-school research course. No student contributed to this code. Catalogued by Null Design as ND-007.

NLP pipeline for measuring political polarity in text corpora.

**This system classifies TEXT CONTENT, not individuals. Do NOT use outputs to infer any individual's political beliefs or any protected demographic trait.**

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline on synthetic data
bash run_all.sh
```

Or step by step:

```bash
# Generate synthetic test data
python -m scripts.generate_synthetic --n-per-class 50

# Preprocess, label, and split
python -m scripts.prepare_data --config configs/default.yaml --skip-lang-filter

# Train baseline model (Logistic Regression on TF-IDF)
python -m scripts.train_baseline --config configs/default.yaml

# Evaluate
python -m scripts.evaluate --config configs/default.yaml --model-type baseline

# Analyze a single text
python -m scripts.infer --model-type baseline \
    --text "The government should invest more in renewable energy."

# Generate subset reports
python -m scripts.report_subsets --model-type baseline --group-by source region

# Launch web UI
python -m web_ui.app
```

## Google Colab

Open `notebooks/political_polarity_colab.ipynb` in Google Colab for a self-contained version you can run in the browser.

## Project Structure

```
political-polarity/
├── configs/
│   └── default.yaml              # Configuration (paths, hyperparameters, k-anonymity)
├── data/
│   ├── raw/                      # Raw JSONL input files
│   ├── processed/                # Preprocessed train/val/test splits
│   └── labels/                   # Outlet bias CSV, human labels CSV
├── polarity/                     # Core library
│   ├── data/
│   │   ├── ingest.py             # JSONL/TXT/HTML ingestion
│   │   ├── preprocess.py         # Dedup, language filter, normalization
│   │   ├── labeling.py           # Outlet-level + human label management
│   │   └── splitting.py          # Stratified + temporal splitting
│   ├── features/
│   │   ├── tfidf_features.py     # TF-IDF vectorization, chi2/MI phrases
│   │   └── keyword_extraction.py # Embedding-aware keyword extraction
│   ├── models/
│   │   ├── baseline.py           # LogReg/SVM classifier
│   │   ├── transformer_model.py  # Fine-tuned transformer (DistilBERT)
│   │   ├── semi_supervised.py    # Tier 3 label expansion
│   │   └── inference.py          # Unified inference pipeline
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, F1, ECE, error analysis
│   │   └── plots.py              # Confusion matrix, calibration, distributions
│   ├── reporting/
│   │   └── subset_analysis.py    # k-anonymity aggregated reports
│   └── utils/
│       ├── config.py             # YAML config loader
│       ├── reproducibility.py    # Seeds, artifact paths
│       └── logging_setup.py      # Logging configuration
├── scripts/                      # CLI entry points
│   ├── generate_synthetic.py     # Create synthetic test data
│   ├── prepare_data.py           # Full data preparation pipeline
│   ├── train_baseline.py         # Train baseline model
│   ├── train_transformer.py      # Train transformer model
│   ├── evaluate.py               # Evaluate with full metrics
│   ├── infer.py                  # Run inference on new documents
│   └── report_subsets.py         # Generate subset reports
├── tests/                        # Unit tests
├── web_ui/
│   └── app.py                    # Gradio web interface
├── notebooks/
│   └── political_polarity_colab.ipynb  # Colab notebook
├── artifacts/                    # Model checkpoints, plots, metrics
├── run_all.sh                    # One-command pipeline run
├── requirements.txt              # Pinned dependencies
├── pyproject.toml                # Project metadata
├── MODEL_CARD.md                 # Ethical guidance and limitations
└── README.md
```

## Data Format

Input documents are JSONL files with this schema:

```json
{
  "doc_id": "article_001",
  "text": "The full text of the article...",
  "title": "Article Title",
  "date": "2024-03-15",
  "source": "outlet_name",
  "url": "https://example.com/article",
  "subset_meta": {
    "region": "Northeast",
    "age_bucket": "25-34"
  }
}
```

Required fields: `doc_id`, `text`. All others are optional.

## Labeling Strategy

### Tier 1: Outlet-level distant supervision

Provide `data/labels/outlet_bias.csv`:

```csv
outlet_name,label
nytimes,left
foxnews,right
reuters,center
```

**Warning:** Outlet label != article ideology. This is weak supervision.

### Tier 2: Human labels

Provide `data/labels/human_labels.csv`:

```csv
doc_id,label
article_001,left
article_002,right
```

Human labels override outlet labels when both exist.

### Tier 3: Semi-supervised expansion

Enable in config (`semi_supervised.enabled: true`). Uses model confidence to propose new labels with strict thresholds.

## Models

### Model A: Baseline (fast, interpretable)

- Logistic Regression on TF-IDF n-grams (1-3)
- Calibrated probabilities
- Top coefficient features per class as explanations

### Model B: Transformer (stronger)

- Fine-tuned DistilBERT for 3-class classification
- Early stopping, warmup scheduling
- Leave-one-out or attention-based interpretability
- CPU-friendly mode (smaller batches, fewer epochs)

```bash
python -m scripts.train_transformer --config configs/default.yaml
python -m scripts.evaluate --model-type transformer
```

## Outputs

### A. Document-level polarity

- P(left), P(center), P(right) probabilities
- Polarity score: P(right) - P(left), range [-1, +1]
- Entropy (uncertainty measure)

### B. Discriminative phrases

- TF-IDF chi-square discriminative phrases per class
- Model coefficient top features per class
- Embedding-aware keyword extraction

### C. Subset reports

- Mean/median polarity by subset (source, region, etc.)
- Class distribution per group
- k-anonymity enforced (groups below threshold are suppressed)

### D. Evaluation

- Accuracy, macro-F1, per-class precision/recall/F1
- Confusion matrix plots
- Calibration curve and ECE
- Topic leakage checks
- Error analysis with misclassified examples

## Configuration

Edit `configs/default.yaml` to adjust:

- Data paths
- Model hyperparameters (C, learning_rate, epochs, etc.)
- TF-IDF parameters (ngram_range, max_features)
- k-anonymity threshold
- Evaluation settings

## Tests

```bash
pytest tests/ -v
```

## Web UI

```bash
python -m web_ui.app
```

Opens a Gradio interface at `http://localhost:7860` where you can paste text and get polarity ratings.

## Privacy Safeguards

- Never infers individual political affiliation
- Never guesses demographic attributes
- k-anonymity: groups below threshold (default 30) are suppressed
- Only uses externally provided metadata for subset analysis
- All analysis is aggregate, never individual-level
