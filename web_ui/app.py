"""Simple Gradio web UI for text polarity rating.

Launch with: python -m web_ui.app
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_app():
    """Create and return the Gradio app."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Install with: pip install gradio")
        print("Falling back to CLI mode.")
        return None

    from polarity.utils.config import load_config
    from polarity.models.inference import PolarityInference

    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    artifacts_dir = config["data"]["artifacts_dir"]

    # Try to load model
    inference = None
    model_path = Path(artifacts_dir) / "baseline_model_latest.pkl"
    vec_path = Path(artifacts_dir) / "tfidf_vectorizer_latest.pkl"

    if model_path.exists() and vec_path.exists():
        inference = PolarityInference(
            model_type="baseline",
            model_path=str(model_path),
            vectorizer_path=str(vec_path),
        )
        logger.info("Loaded baseline model for web UI")
    else:
        logger.warning("No trained model found. Run training first.")

    def analyze_text(text: str) -> str:
        """Analyze text and return formatted results."""
        if not text or not text.strip():
            return "Please enter some text to analyze."

        if inference is None:
            return (
                "No model available. Please train a model first:\n"
                "  python -m scripts.generate_synthetic\n"
                "  python -m scripts.prepare_data --skip-lang-filter\n"
                "  python -m scripts.train_baseline"
            )

        result = inference.score_text(text)

        # Build visual output
        lines = []
        lines.append("## Polarity Analysis Results\n")
        lines.append(f"**Predicted Label:** {result['predicted_label'].upper()}\n")
        lines.append(f"**Polarity Score:** {result['polarity_score']:+.3f} "
                      f"(Left -1.0 ← → +1.0 Right)\n")

        lines.append("\n### Class Probabilities\n")
        for cls, prob in result["probabilities"].items():
            bar_len = int(prob * 30)
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            lines.append(f"- **{cls:>8}**: `{prob:.3f}` {bar}")

        lines.append(f"\n**Entropy:** {result['entropy']:.3f} "
                      "(lower = more confident)\n")

        # Interpretability
        try:
            importance = inference.get_interpretability(text)
            if importance:
                lines.append("\n### Top Contributing Features\n")
                lines.append("| Feature | Score |")
                lines.append("|---------|-------|")
                for feat, score in importance[:10]:
                    direction = "\u2192 right" if score > 0 else "\u2190 left"
                    lines.append(f"| {feat} | {score:+.4f} ({direction}) |")
        except Exception:
            pass

        lines.append("\n---")
        lines.append("\n> **WARNING:** This classifies TEXT CONTENT, not the author. "
                      "Do NOT use to infer any individual's political affiliation.")

        return "\n".join(lines)

    # Build Gradio interface
    with gr.Blocks(
        title="Political Polarity Analyzer",
    ) as app:
        gr.Markdown(
            "# Political Polarity Text Analyzer\n"
            "Paste text below to analyze its political polarity.\n\n"
            "> **Ethics Notice:** This tool classifies TEXT CONTENT only. "
            "It does NOT determine any individual's political beliefs. "
            "See the Model Card for limitations and ethical guidance."
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Input Text",
                    placeholder="Paste an article, transcript, or opinion piece here...",
                    lines=12,
                    max_lines=30,
                )
                analyze_btn = gr.Button("Analyze Polarity", variant="primary")

            with gr.Column(scale=1):
                output = gr.Markdown(label="Results")

        analyze_btn.click(
            fn=analyze_text,
            inputs=[text_input],
            outputs=[output],
        )

        # Example texts
        gr.Examples(
            examples=[
                ["The government should increase funding for public healthcare and expand social safety net programs to help working families."],
                ["Lower taxes and reduced regulation are the keys to economic growth. Small businesses need relief from bureaucratic red tape."],
                ["The debate over this policy involves legitimate perspectives on both sides. Finding common ground remains a challenge."],
            ],
            inputs=[text_input],
            label="Example Texts (click to load)",
        )

        gr.Markdown(
            "---\n"
            "**Model:** Logistic Regression on TF-IDF n-grams | "
            "**Classes:** Left, Center, Right | "
            "[Model Card](../MODEL_CARD.md)"
        )

    return app


def main():
    app = create_app()
    if app is not None:
        import gradio as gr
        app.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860,
            theme=gr.themes.Soft(),
        )
    else:
        print("Cannot launch web UI without Gradio.")


if __name__ == "__main__":
    main()
