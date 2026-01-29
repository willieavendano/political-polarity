"""Generate synthetic dataset for smoke testing.

IMPORTANT: This data is ENTIRELY SYNTHETIC and FICTIONAL.
It is designed only to verify the pipeline runs end-to-end.
The "political" content is intentionally simplistic and stereotyped
for testing purposes. Do NOT draw any conclusions from model
performance on this synthetic data.
"""

import json
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Synthetic article templates by class
LEFT_TEMPLATES = [
    "The government should increase funding for public healthcare programs to ensure universal coverage for all citizens. Expanding Medicaid and supporting community health centers is essential.",
    "Climate change is the defining challenge of our generation. We need aggressive action to reduce carbon emissions, invest in renewable energy, and transition away from fossil fuels.",
    "Income inequality has reached alarming levels. Progressive taxation and raising the minimum wage would help create a more equitable society for working families.",
    "Gun violence is a public health crisis. Common-sense gun safety legislation including universal background checks is urgently needed to protect communities.",
    "Investments in public education, including universal pre-K and affordable college tuition, are crucial for the future economic competitiveness of our nation.",
    "Workers deserve stronger protections and the right to organize. Union membership has declined while corporate profits have soared, widening the economic divide.",
    "Immigration reform should provide a pathway to citizenship for undocumented immigrants who contribute to our economy and communities every day.",
    "Social safety net programs like SNAP and unemployment insurance are vital investments that help families during difficult times and stimulate the economy.",
    "Criminal justice reform must address systemic inequities. We need to end mass incarceration and invest in rehabilitation and community-based alternatives.",
    "Reproductive rights are fundamental healthcare decisions that should be made by individuals and their doctors, not politicians in state capitols.",
]

RIGHT_TEMPLATES = [
    "Lower taxes and reduced government regulation are the keys to economic growth. Small businesses are the backbone of America and need relief from bureaucratic red tape.",
    "The Second Amendment clearly protects the individual right to keep and bear arms. Any attempt to restrict lawful gun ownership infringes on constitutional rights.",
    "Securing our borders is a matter of national sovereignty. We must enforce existing immigration laws and build the infrastructure needed to prevent illegal crossings.",
    "Free market solutions, not government mandates, are the most effective way to drive innovation in healthcare and bring down costs for consumers.",
    "Traditional family values and religious liberty are the foundation of a strong society. Government should not force individuals to act against their deeply held beliefs.",
    "A strong national defense is essential for protecting American interests abroad. Military readiness must remain a top budget priority against growing threats.",
    "School choice empowers parents to select the best educational environment for their children. Charter schools and voucher programs promote competition and excellence.",
    "The national debt is a threat to future generations. Fiscal discipline and reducing wasteful government spending must be priorities for responsible governance.",
    "Energy independence through domestic oil, gas, and coal production creates jobs and strengthens our national security position against adversarial nations.",
    "Law enforcement officers deserve our support and respect. Defunding police departments would make communities less safe and embolden criminal activity.",
]

CENTER_TEMPLATES = [
    "The debate over healthcare policy involves legitimate perspectives on both sides. Finding common ground between market-based approaches and public programs remains challenging.",
    "Environmental policy must balance ecological protection with economic realities. Both renewable energy investment and energy sector job preservation deserve consideration.",
    "The federal budget reflects competing priorities. Defense spending, social programs, and infrastructure investment all have merit depending on the specific proposals.",
    "Education reform efforts have produced mixed results across different states and districts. More research is needed to identify which approaches work best.",
    "Trade policy involves complex tradeoffs between consumer prices, domestic manufacturing jobs, and international relationships that resist simple solutions.",
    "Immigration policy discussions often generate more heat than light. A comprehensive approach addressing border security and legal immigration pathways is worth exploring.",
    "The relationship between government regulation and economic growth is complex and context-dependent, varying significantly across industries and time periods.",
    "Technology policy raises important questions about privacy, innovation, and competition that don't map neatly onto traditional political categories.",
    "Infrastructure investment enjoys bipartisan support in principle, though debates over funding mechanisms and project prioritization remain contentious.",
    "Addressing the opioid crisis requires cooperation across party lines, combining law enforcement approaches with public health interventions and treatment funding.",
]

OUTLETS = {
    "left": ["progressive_daily", "liberal_times", "forward_press"],
    "center": ["national_gazette", "balanced_news", "independent_journal"],
    "right": ["conservative_herald", "liberty_post", "patriot_review"],
}

REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def generate_synthetic_dataset(
    n_per_class: int = 50,
    output_dir: str = "./data/raw",
    seed: int = 42,
) -> None:
    """Generate synthetic JSONL dataset.

    Args:
        n_per_class: Number of documents per class.
        output_dir: Output directory for JSONL file.
        seed: Random seed.
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    base_date = datetime(2024, 1, 1)

    for cls, templates in [
        ("left", LEFT_TEMPLATES),
        ("center", CENTER_TEMPLATES),
        ("right", RIGHT_TEMPLATES),
    ]:
        outlets = OUTLETS[cls]
        for i in range(n_per_class):
            template = templates[i % len(templates)]
            # Add some variation
            prefix = random.choice([
                "Analysis: ", "Opinion: ", "Report: ", "Commentary: ", "",
            ])
            suffix = random.choice([
                " This issue will likely be debated in the coming months.",
                " Experts continue to weigh in on this topic.",
                " The public remains divided on this question.",
                "",
            ])
            text = prefix + template + suffix

            outlet = random.choice(outlets)
            date = base_date + timedelta(days=random.randint(0, 365))
            region = random.choice(REGIONS)
            age_bucket = random.choice(AGE_BUCKETS)

            record = {
                "doc_id": f"synth_{cls}_{i:04d}",
                "text": text,
                "title": f"Synthetic {cls} article {i}",
                "date": date.strftime("%Y-%m-%d"),
                "source": outlet,
                "url": f"https://example.com/{outlet}/article_{i}",
                "subset_meta": {
                    "region": region,
                    "age_bucket": age_bucket,
                    "synthetic": True,
                },
            }
            records.append(record)

    # Shuffle
    random.shuffle(records)

    # Write JSONL
    output_path = output_dir / "synthetic_corpus.jsonl"
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Generated %d synthetic documents to %s", len(records), output_path)

    # Also create outlet bias file
    labels_dir = Path("./data/labels")
    labels_dir.mkdir(parents=True, exist_ok=True)

    bias_path = labels_dir / "outlet_bias.csv"
    with open(bias_path, "w") as f:
        f.write("outlet_name,label\n")
        for label, outlets in OUTLETS.items():
            for outlet in outlets:
                f.write(f"{outlet},{label}\n")

    logger.info("Generated outlet bias file: %s", bias_path)

    # Summary
    logger.info("\n=== SYNTHETIC DATA SUMMARY ===")
    logger.info("Total documents: %d", len(records))
    logger.info("Per class: %d", n_per_class)
    logger.info("Outlets: %s", {k: v for k, v in OUTLETS.items()})
    logger.info("WARNING: This is SYNTHETIC data for testing only!")
    logger.info("Do NOT draw conclusions from model results on this data.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for smoke testing"
    )
    parser.add_argument(
        "--n-per-class", type=int, default=50,
        help="Number of documents per class (default: 50)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/raw",
        help="Output directory for JSONL (default: ./data/raw)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    generate_synthetic_dataset(
        n_per_class=args.n_per_class,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
