# From Data Quality to Data Value

**A discourse-level audit pipeline for quality-filtered LLM pretraining corpora.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/llm-discourse-audit/blob/main/notebooks/01_colab_quickstart.ipynb)

> Replace `YOUR_GITHUB_USERNAME` in the badge URL after uploading this repository to GitHub.

## Project aim

This repository supports the project:

**From Data Quality to Data Value: A Discourse-Level Audit of Quality-Filtered LLM Pretraining Corpora**

The first version focuses on comparing quality-filtered and less explicitly quality-filtered public pretraining corpora. Instead of training a large language model, the pipeline audits what kinds of communicative data are retained or reduced by filtering.

The current GitHub-ready version implements the first two steps:

1. **Stream and sample public corpora** from Hugging Face.
2. **Extract transparent discourse, linguistic, and governance proxy features.**

## Research question

When pretraining data is filtered for "quality" or "educational value", does the corpus also shift in genre, register, discourse function, and interactional structure?

The broader claim is:

> Data quality is not identical to data value.

A dataset can become more educational or benchmark-friendly while losing other types of communicative value, such as informal language, dialogue-like data, personal narrative, troubleshooting discourse, question-answering, and long-tail registers.

## Current datasets

The default config samples from:

| Corpus | Hugging Face dataset | Default config | Purpose |
|---|---|---|---|
| FineWeb | `HuggingFaceFW/fineweb` | `sample-10BT` | less explicitly educational-filtered web baseline |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | educational-quality-filtered comparison |

Edit `configs/datasets.yaml` to change `n_docs`.

## Repository structure

```text
llm-discourse-audit/
├── configs/
│   ├── datasets.yaml
│   └── labels.yaml
├── data/
│   ├── raw/
│   ├── features/
│   └── processed/
├── docs/
├── notebooks/
│   └── 01_colab_quickstart.ipynb
├── outputs/
│   ├── tables/
│   └── figures/
├── scripts/
│   └── run_steps_1_2.sh
├── src/
│   ├── 00_stream_sample.py
│   └── 01_extract_features.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

The `data/` and `outputs/` folders are intentionally excluded from Git tracking except for `.gitkeep` files.

## Run in Colab

1. Upload this repository to GitHub.
2. Open `README.md` on GitHub.
3. Replace the Colab badge URL:

```text
YOUR_GITHUB_USERNAME
```

with your actual GitHub username.
4. Click **Open in Colab**.
5. In the first notebook cell, also edit:

```python
GITHUB_USERNAME = "YOUR_GITHUB_USERNAME"
```

Then run all cells.

## Run locally

Create an environment:

```bash
conda create -n discourse-audit python=3.11 -y
conda activate discourse-audit
pip install -r requirements.txt
```

Run the first two steps:

```bash
bash scripts/run_steps_1_2.sh
```

Or run them manually:

```bash
python src/00_stream_sample.py \
  --only fineweb fineweb_edu \
  --config configs/datasets.yaml \
  --out_dir data/raw \
  --seed 42

python src/01_extract_features.py \
  --raw_dir data/raw \
  --out data/features/features.parquet
```

## Step 1 output

After sampling, you should see:

```text
data/raw/fineweb.jsonl.gz
data/raw/fineweb_edu.jsonl.gz
```

Each line is one JSON document with fields such as:

```json
{
  "doc_id": "fineweb_xxxxx",
  "corpus": "fineweb",
  "dataset_id": "HuggingFaceFW/fineweb",
  "config": "sample-10BT",
  "split": "train",
  "text": "..."
}
```

## Step 2 output

After feature extraction, you should see:

```text
data/features/features.parquet
```

Important feature groups:

| Feature group | Examples |
|---|---|
| Length | `char_count`, `word_count`, `sentence_count` |
| Lexical diversity | `type_token_ratio`, `word_entropy_first_2000` |
| Subjectivity / interactionality | `first_person_rate`, `second_person_rate`, `question_per_1k_words` |
| Discourse structure | `dialogue_like_line_rate`, `list_marker_line_rate`, `imperative_proxy_line_rate` |
| Governance proxies | `pii_risk`, `boilerplate_count`, `url_count`, `code_flag` |

## Recommended first run

Use the default:

```yaml
n_docs: 5000
```

After the pipeline succeeds, increase to:

```yaml
n_docs: 20000
```

Then later:

```yaml
n_docs: 50000
```

Do not start with 100k per corpus in Colab until the smaller run works.

## Next planned steps

The next scripts to add are:

1. `02_classify_discourse.py`  
   Genre/register/discourse classification.

2. `03_compute_metrics.py`  
   Entropy, Jensen-Shannon distance, log odds, and numeric feature comparisons.

3. `04_plot_results.py`  
   Figures for the paper.

4. `05_manual_audit_sample.py`  
   A researcher-coded reliability sheet for validating automatic labels.

## Paper framing

Possible title:

**From Data Quality to Data Value: A Discourse-Level Audit of Quality-Filtered LLM Pretraining Corpora**

Possible contribution statement:

1. We introduce a discourse-level audit framework for public LLM pretraining corpora.
2. We compare quality-filtered and less explicitly quality-filtered web corpora.
3. We show that filtering for quality can reshape the communicative profile of data, not merely remove noise.
4. We argue for multi-objective data value governance rather than scalar quality filtering.

## Notes

This repository does not include sampled raw text or generated outputs. They are intentionally ignored by Git because corpus samples can become large.
