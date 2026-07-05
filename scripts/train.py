"""Train a multilingual student encoder via knowledge distillation.

Downloads parallel sentence data from OPUS-100 for the target languages,
loads the frozen English teacher and multilingual student, and trains the
student to match the teacher's embeddings.

Usage:
    python scripts/train.py
    python scripts/train.py --languages fr pt es --n-samples 5000 --epochs 3
"""
import argparse
import logging

from torch.utils.data import DataLoader, ConcatDataset

from mse.data.loader import load_parallel_sentences
from mse.data.parallel_dataset import ParallelSentenceDataset
from mse.models.student import StudentEncoder
from mse.models.teacher import TeacherEncoder
from mse.training.config import DistillationConfig
from mse.training.distillation_trainer import DistillationTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages",   type=str, nargs="+",
                        default=["de", "fr", "es", "pt", "it"])
    parser.add_argument("--n-samples",   type=int, default=5000,
                        help="Number of parallel sentence pairs per language.")
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def build_datasets(languages, n_samples, teacher_tokenizer, student_tokenizer, max_seq_len):
    """Load parallel data for each language and combine into one dataset.

    A 90/10 split is taken within each language before combining, so the
    validation set contains held-out sentences from every language, not
    just one.
    """
    train_datasets = []
    val_datasets = []

    for lang in languages:
        logger.info(f"Loading {lang}...")
        english, translated = load_parallel_sentences(lang, n_samples)

        n_val = max(1, int(0.1 * len(english)))
        train_english, val_english = english[:-n_val], english[-n_val:]
        train_translated, val_translated = translated[:-n_val], translated[-n_val:]

        train_datasets.append(ParallelSentenceDataset(
            train_english, train_translated, teacher_tokenizer, student_tokenizer, max_seq_len
        ))
        val_datasets.append(ParallelSentenceDataset(
            val_english, val_translated, teacher_tokenizer, student_tokenizer, max_seq_len
        ))

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets)


def main():
    args = parse_args()

    logger.info("Loading teacher (all-MiniLM-L6-v2)...")
    teacher = TeacherEncoder()

    logger.info("Loading student (xlm-roberta-base)...")
    student = StudentEncoder(teacher_dim=teacher.embedding_dim)

    # The teacher's own tokenizer, accessed via the underlying SentenceTransformer
    teacher_tokenizer = teacher._st_model.tokenizer

    train_dataset, val_dataset = build_datasets(
        args.languages, args.n_samples, teacher_tokenizer,
        student.tokenizer, args.max_seq_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size)

    logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    logger.info(f"Student parameters: {student.parameter_count():,}")

    config = DistillationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.max_seq_len,
        languages=tuple(args.languages),
        seed=args.seed,
    )

    trainer = DistillationTrainer(student, teacher, train_loader, val_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
