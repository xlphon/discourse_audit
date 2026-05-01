# Research Plan

## Working title

From Data Quality to Data Value: A Discourse-Level Audit of Quality-Filtered LLM Pretraining Corpora

## Core comparison

- FineWeb
- FineWeb-Edu

## Key hypothesis

Educational or quality-oriented filtering changes not only the perceived quality of the data, but also the communicative structure of the corpus.

Expected shifts:

- More educational/tutorial/explanatory prose
- More formal and instructional register
- Less dialogue-like structure
- Less first-person/personal narrative
- Fewer questions and Q&A-like documents
- Less informal or conversational language

## First-stage features

The first two scripts extract transparent features only. These are deliberately simple and interpretable.

Important dependent measures:

- `first_person_rate`
- `second_person_rate`
- `question_per_1k_words`
- `dialogue_like_line_rate`
- `imperative_proxy_line_rate`
- `type_token_ratio`
- `word_entropy_first_2000`
- `boilerplate_count`
- `pii_risk`
- `code_flag`

## Next step

Add a discourse classifier and validate it on a small researcher-coded sample.
