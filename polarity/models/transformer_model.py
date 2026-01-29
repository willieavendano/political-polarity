"""Transformer-based model for 3-class polarity classification."""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class PolarityDataset(Dataset):
    """PyTorch dataset for polarity classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class TransformerPolarityModel:
    """Fine-tuned transformer for 3-class polarity classification.

    Supports CPU-friendly configuration for environments without GPU.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 3,
        class_names: list[str] | None = None,
        max_length: int = 512,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.num_classes = num_classes
        self.class_names = class_names or ["left", "center", "right"]
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Transformer device: %s", self.device)

    def _load_pretrained(self) -> None:
        """Load pretrained model and tokenizer."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
        )
        self.model.to(self.device)
        logger.info("Loaded pretrained model: %s", self.model_name)

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 5,
        early_stopping_patience: int = 2,
        warmup_ratio: float = 0.1,
        cpu_friendly: bool = True,
        cpu_batch_size: int = 4,
        cpu_num_epochs: int = 2,
    ) -> dict:
        """Train the transformer model with early stopping.

        Args:
            train_texts: Training texts.
            train_labels: Training labels (integers).
            val_texts: Validation texts.
            val_labels: Validation labels.
            batch_size: Batch size (GPU).
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            num_epochs: Number of epochs (GPU).
            early_stopping_patience: Stop if val loss doesn't improve.
            warmup_ratio: Fraction of steps for LR warmup.
            cpu_friendly: Use CPU-friendly settings if on CPU.
            cpu_batch_size: Smaller batch size for CPU.
            cpu_num_epochs: Fewer epochs for CPU.

        Returns:
            Training history dict with losses and metrics.
        """
        self._load_pretrained()

        # Adjust for CPU
        if self.device.type == "cpu" and cpu_friendly:
            batch_size = cpu_batch_size
            num_epochs = cpu_num_epochs
            logger.info(
                "CPU mode: batch_size=%d, num_epochs=%d", batch_size, num_epochs
            )

        # Create datasets
        train_dataset = PolarityDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = PolarityDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # LR scheduler with warmup
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(num_epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train_loss)

            # Validate
            val_loss, val_acc = self._evaluate_epoch(val_loader)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            logger.info(
                "Epoch %d/%d: train_loss=%.4f, val_loss=%.4f, val_acc=%.4f",
                epoch + 1, num_epochs, avg_train_loss, val_loss, val_acc,
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            logger.info("Restored best model (val_loss=%.4f)", best_val_loss)

        return history

    def _evaluate_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Predict class probabilities.

        Args:
            texts: Input texts.
            batch_size: Batch size.

        Returns:
            Array of shape (n_samples, 3) with probabilities.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()
        all_probs = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encodings = {k: v.to(self.device) for k, v in encodings.items()}

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def predict(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(texts, batch_size)
        return np.argmax(probs, axis=1)

    def polarity_score(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Compute polarity score: P(right) - P(left)."""
        probs = self.predict_proba(texts, batch_size)
        return probs[:, 2] - probs[:, 0]

    def get_token_importance(
        self,
        text: str,
        method: str = "leave_one_out",
        top_k: int = 15,
    ) -> list[tuple[str, float]]:
        """Get token-level importance scores for interpretability.

        Args:
            text: Input text.
            method: 'leave_one_out' or 'attention'.
            top_k: Number of top tokens to return.

        Returns:
            List of (token, importance_score) tuples.
        """
        if method == "leave_one_out":
            return self._leave_one_out_importance(text, top_k)
        elif method == "attention":
            return self._attention_importance(text, top_k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _leave_one_out_importance(
        self, text: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Leave-one-out token importance.

        For each token, measures how much the predicted class probability
        changes when that token is removed.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()

        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return []

        # Full prediction
        full_probs = self.predict_proba([text])[0]
        predicted_class = int(np.argmax(full_probs))
        baseline_prob = full_probs[predicted_class]

        # Leave one out
        importances = []
        for i, token in enumerate(tokens[:100]):  # limit for speed
            reduced_tokens = tokens[:i] + tokens[i + 1:]
            reduced_text = self.tokenizer.convert_tokens_to_string(reduced_tokens)

            if not reduced_text.strip():
                continue

            reduced_probs = self.predict_proba([reduced_text])[0]
            importance = baseline_prob - reduced_probs[predicted_class]
            importances.append((token, float(importance)))

        importances.sort(key=lambda x: abs(x[1]), reverse=True)
        return importances[:top_k]

    def _attention_importance(
        self, text: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Attention-weight based importance (approximate).

        Averages attention weights across heads and layers for each token.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**encoding, output_attentions=True)

        # Average attention across layers and heads
        # attentions: tuple of (batch, heads, seq, seq)
        attentions = outputs.attentions
        avg_attention = torch.stack(attentions).mean(dim=[0, 1, 2])  # (seq_len,)
        avg_attention = avg_attention.squeeze().cpu().numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"].squeeze().tolist()
        )

        # Pair tokens with attention scores, skip special tokens
        token_scores = []
        for token, score in zip(tokens, avg_attention):
            if token in ("[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"):
                continue
            token_scores.append((token, float(score)))

        token_scores.sort(key=lambda x: x[1], reverse=True)
        return token_scores[:top_k]

    def save(self, path: str | Path) -> None:
        """Save model and tokenizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        # Save metadata
        import json
        meta = {
            "class_names": self.class_names,
            "model_name": self.model_name,
            "max_length": self.max_length,
        }
        with open(path / "polarity_meta.json", "w") as f:
            json.dump(meta, f)
        logger.info("Saved transformer model to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "TransformerPolarityModel":
        """Load saved model."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import json

        path = Path(path)
        meta_path = path / "polarity_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {"class_names": ["left", "center", "right"], "max_length": 512}

        instance = cls(
            model_name=str(path),
            class_names=meta.get("class_names", ["left", "center", "right"]),
            max_length=meta.get("max_length", 512),
            device=device,
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        instance.model.to(instance.device)
        logger.info("Loaded transformer model from %s", path)
        return instance
