# Multilingual Sentence Embeddings

A from-scratch implementation of multilingual sentence embedding via **knowledge distillation**, based on [Reimers & Gurevych (2020), *Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation*](https://arxiv.org/abs/2004.09813).

A frozen English sentence encoder (teacher) guides a multilingual model (student) to produce aligned embeddings across languages. After training, sentences with the same meaning in different languages map to nearby vectors, enabling cross-lingual retrieval with no target-language labels, only parallel translation data.

---

## Results

**Cross-lingual retrieval** — given a sentence in the target language, retrieve its correct English translation from a pool of 500 candidates by cosine similarity.

| Language | P@1 | P@5 | P@10 | MRR |
|---|---|---|---|---|
| German | 0.722 | 0.876 | 0.910 | 0.788 |
| French | 0.862 | 0.950 | 0.960 | 0.902 |
| Spanish | 0.846 | 0.930 | 0.952 | 0.887 |
| Portuguese | 0.820 | 0.896 | 0.924 | 0.856 |
| Italian | 0.802 | 0.904 | 0.926 | 0.852 |
| **Average** | **0.810** | **0.911** | **0.934** | **0.857** |

Trained on 8,000 parallel sentence pairs per language (40,000 total), 5 epochs, on CPU. Given a sentence in any of the five languages, the correct English translation is the top retrieved result over 80% of the time, and is in the top 5 over 91% of the time.

**Effect of more data and training**

| Configuration | Avg P@1 | Avg MRR |
|---|---|---|
| 2,000 samples/language, 3 epochs | 0.414 | 0.539 |
| **8,000 samples/language, 5 epochs** | **0.810** | **0.857** |

Scaling data 4x and training 2 additional epochs nearly doubled retrieval accuracy — the single largest lever, consistent with a common pattern in applied ML where more data and training time outweighs architectural tuning at this scale.

---

## How it works

### The problem

Training a multilingual embedding model from scratch requires enormous multilingual data and compute. Most of that cost is unnecessary if a strong monolingual (English) model already exists.

### The idea — knowledge distillation

1. A frozen **teacher** model (a pretrained English sentence encoder) embeds English sentences
2. A trainable **student** model (a multilingual transformer) embeds the same sentences translated into another language
3. The student is trained to match the teacher's embedding for the same meaning, using MSE loss
4. Only the student's weights update — the teacher never changes

After training, the student places semantically equivalent sentences from any trained language near the teacher's English embedding space, enabling direct cross-lingual comparison.

### Why this works better than training multilingual from scratch

The student (XLM-RoBERTa) already has cross-lingual structure from its own pretraining. Distillation doesn't teach it languages from nothing; it aligns its existing multilingual knowledge to a specific target embedding space, which is a much easier learning problem than training alignment from raw data alone.

---

## Project structure

```
src/mse/
├── models/
│   ├── base.py               # Abstract SentenceEncoder interface
│   ├── teacher.py            # Frozen English encoder (all-MiniLM-L6-v2)
│   └── student.py            # Trainable multilingual encoder (XLM-RoBERTa)
├── data/
│   ├── loader.py              # OPUS-100 parallel sentence loading
│   └── parallel_dataset.py    # PyTorch Dataset for (English, translated) pairs
├── training/
│   ├── config.py                  # DistillationConfig dataclass
│   └── distillation_trainer.py    # Training loop, MSE loss, LR scheduling
└── evaluation/
    └── metrics.py             # Precision@k, Mean Reciprocal Rank

scripts/
├── train.py       # Run distillation training
└── evaluate.py    # Cross-lingual retrieval evaluation

tests/             # Mirrors src/ structure, 15 tests
```

---

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python scripts/train.py --languages de fr es pt it --n-samples 8000 --epochs 5 --batch-size 8 --max-seq-len 64
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_5.pt --n-eval 500
python -m pytest
```

---

## Key implementation details

**Frozen teacher, trainable student** (`src/mse/training/distillation_trainer.py`): The teacher's forward pass runs under `torch.no_grad()` and its parameters have `requires_grad=False` set at construction. Only the student receives gradient updates.

**Mean pooling** (`src/mse/models/student.py`): Sentence embeddings are computed by averaging token embeddings, masked to ignore padding — the standard approach in sentence-transformers, generally outperforming CLS-token pooling.

**Dimension projection**: The student's native hidden size (768 for XLM-R-base) is projected down to the teacher's embedding dimension (384 for MiniLM) via a learned linear layer, so the MSE loss compares vectors in a common space.

**Language pair naming inconsistency** (`src/mse/data/loader.py`): OPUS-100 names its configs inconsistently — some are `en-X`, others `X-en`, depending on alphabetical ordering of the codes. The loader tries both automatically.

**CPU memory constraints**: Training a 278M-parameter model on CPU is memory-intensive. Batch size 8 and max sequence length 64 were chosen specifically to avoid swap-thrashing observed at larger batch sizes (32) and longer sequences (128) on a 16GB machine.

---

## Requirements

- Python 3.10+
- PyTorch 2.2+
- transformers, sentence-transformers, datasets
- NumPy, SciPy, Matplotlib, tqdm
