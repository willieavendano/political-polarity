# Model Card: Political Polarity Text Classifier

## Model Details

- **Model type**: Text classification (3-class: left, center, right)
- **Baseline**: Logistic Regression on TF-IDF n-gram features
- **Stronger model**: Fine-tuned DistilBERT
- **Training data**: User-provided labeled text corpus (outlet-level distant supervision or human labels)
- **Language**: English
- **Political context**: United States

## Intended Use

### Primary use cases

- **Media research**: Studying political framing and language patterns across outlets
- **Corpus analysis**: Understanding the distribution of political perspectives in a text collection
- **Linguistic research**: Identifying phrases and language patterns associated with political perspectives

### Out-of-scope uses (DO NOT USE FOR)

- Inferring any individual's political beliefs, preferences, or affiliation
- Targeting, profiling, or discriminating against individuals or groups
- Automated content moderation, censorship, or suppression decisions
- Hiring, admissions, lending, or any consequential decisions about individuals
- Surveillance or monitoring of individuals' political views
- Creating targeted political advertising or persuasion campaigns

## Limitations

### Data limitations

- **Outlet != article**: An outlet labeled "left" may publish center or right-leaning articles. Distant supervision using outlet-level labels is inherently noisy.
- **US-centric**: The left/center/right taxonomy reflects a US political framework. It does not generalize to other countries' political spectra.
- **Temporal drift**: Political language and party positions evolve. A model trained on 2024 data may not be accurate for 2020 or 2028 content.
- **Selection bias**: The training corpus likely overrepresents certain topics, outlets, or perspectives.

### Model limitations

- **Topic-ideology conflation**: Discussing a topic (e.g., healthcare, guns) is NOT the same as advocating for a political position. The model may learn spurious topic-ideology correlations.
- **3-class simplification**: The left/center/right scheme is a drastic simplification. Real political views are multidimensional, inconsistent, and context-dependent.
- **Calibration**: Predicted probabilities may not reflect true confidence, especially on out-of-distribution text.
- **Adversarial fragility**: The model can be easily fooled by mixing left and right language, using irony/sarcasm, or discussing topics in unusual ways.

### Measurement limitations

- **Polarity score**: The continuous score (P(right) - P(left)) is a model output, not a ground truth measurement of political lean. Treat it as a rough signal with significant uncertainty.
- **Aggregate statistics**: Even aggregate statistics can be misleading if the underlying model is systematically biased.
- **Entropy**: Low entropy does not guarantee correctness; it means the model is confident, which can happen even on wrong predictions.

## Ethical Considerations

### Risk of harm

1. **Individual profiling**: The greatest risk is using model outputs to label individuals. This is explicitly prohibited. Text polarity != author's beliefs.
2. **Stereotyping groups**: Aggregate statistics about subsets (e.g., by region) may reinforce stereotypes. Always present with appropriate uncertainty and context.
3. **Bias amplification**: The model may amplify biases present in training data, particularly if certain topics or perspectives are underrepresented.
4. **Misuse for targeting**: Polarity scores could theoretically be used to target populations with tailored political content. This is an unethical use.

### Safeguards implemented

- k-anonymity enforcement (minimum group size) for subset reporting
- No individual-level outputs in subset reports
- No demographic inference from text
- Explicit warnings in all outputs
- Topic leakage detection
- Calibration monitoring

### Recommendations for users

1. Always report model limitations alongside results
2. Never present polarity scores as ground truth
3. Validate findings with domain experts
4. Use multiple methods and data sources for triangulation
5. Monitor for temporal drift and retrain as needed
6. Review the topic leakage check to ensure the model is not just learning topics

## Evaluation

Evaluation should include (all implemented):

- Stratified train/val/test split (by label and source)
- Macro-F1 (handles class imbalance)
- Per-class precision, recall, F1
- Confusion matrix
- Expected Calibration Error (ECE)
- Topic leakage analysis
- Error analysis on misclassified examples
- Optional temporal split evaluation

## Citation

If using this system in research, please describe its limitations and cite this model card.
