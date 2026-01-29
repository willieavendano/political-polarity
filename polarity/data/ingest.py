"""Data ingestion: load JSONL documents, convert raw files to JSONL."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# Expected JSONL schema fields
REQUIRED_FIELDS = {"doc_id", "text"}
OPTIONAL_FIELDS = {"title", "date", "source", "url", "subset_meta"}


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame.

    Args:
        path: Path to .jsonl file.

    Returns:
        DataFrame with document records.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d in %s", line_num, path)
                continue

            missing = REQUIRED_FIELDS - set(record.keys())
            if missing:
                logger.warning(
                    "Skipping record at line %d: missing fields %s", line_num, missing
                )
                continue

            records.append(record)

    df = pd.DataFrame(records)
    logger.info("Loaded %d documents from %s", len(df), path)
    return df


def load_jsonl_dir(directory: str | Path) -> pd.DataFrame:
    """Load all JSONL files from a directory.

    Args:
        directory: Path to directory containing .jsonl files.

    Returns:
        Combined DataFrame.
    """
    directory = Path(directory)
    frames = []
    for jsonl_file in sorted(directory.glob("*.jsonl")):
        frames.append(load_jsonl(jsonl_file))

    if not frames:
        logger.warning("No JSONL files found in %s", directory)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total documents from %s", len(combined), directory)
    return combined


def raw_text_to_jsonl(
    input_dir: str | Path,
    output_path: str | Path,
    default_source: str = "unknown",
) -> int:
    """Convert a folder of raw .txt files to JSONL format.

    Each file becomes one document. The filename (without extension)
    is used as the doc_id.

    Args:
        input_dir: Directory containing .txt files.
        output_path: Output .jsonl file path.
        default_source: Default source label if not inferrable.

    Returns:
        Number of documents converted.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for txt_file in sorted(input_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            doc_id = txt_file.stem

            record = {
                "doc_id": doc_id,
                "text": text,
                "title": "",
                "date": "",
                "source": default_source,
                "url": "",
                "subset_meta": {},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Converted %d .txt files to %s", count, output_path)
    return count


def raw_html_to_jsonl(
    input_dir: str | Path,
    output_path: str | Path,
    default_source: str = "unknown",
) -> int:
    """Convert a folder of raw .html files to JSONL format.

    Uses BeautifulSoup for basic text extraction.

    Args:
        input_dir: Directory containing .html files.
        output_path: Output .jsonl file path.
        default_source: Default source label.

    Returns:
        Number of documents converted.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("beautifulsoup4 required for HTML ingestion. pip install beautifulsoup4")
        return 0

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for html_file in sorted(input_dir.glob("*.html")):
            raw = html_file.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(raw, "html.parser")

            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator=" ", strip=True)
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else ""

            doc_id = html_file.stem
            record = {
                "doc_id": doc_id,
                "text": text,
                "title": title,
                "date": "",
                "source": default_source,
                "url": "",
                "subset_meta": {},
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Converted %d .html files to %s", count, output_path)
    return count
