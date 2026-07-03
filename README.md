# Multilingual Sentence Embeddings

A from-scratch implementation of multilingual sentence embedding via **knowledge distillation**, based on [Reimers & Gurevych (2020)](https://arxiv.org/abs/2004.09813).

A frozen English sentence encoder (teacher) guides a multilingual model (student) to produce aligned embeddings across languages. After training, sentences with the same meaning in different languages map to nearby vectors — enabling cross-lingual retrieval with no target-language supervision.

## Languages

German, French, Spanish, Portuguese, Italian → English

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/download_data.py
python scripts/train.py
python scripts/evaluate.py
```
