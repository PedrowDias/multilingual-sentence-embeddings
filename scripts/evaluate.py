"""Evaluate cross-lingual retrieval performance of the trained student.

For each language, loads held-out parallel sentences, embeds the non-English
side with the student and the English side with the teacher, then measures
how often the correct translation is retrieved by cosine similarity.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_epoch_5.pt --languages fr pt
"""
import argparse
import logging
from pathlib import Path

import torch

from mse.data.loader import load_parallel_sentences
from mse.evaluation.metrics import precision_at_k, mean_reciprocal_rank
from mse.models.student import StudentEncoder
from mse.models.teacher import TeacherEncoder

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/checkpoint_epoch_5.pt"))
    parser.add_argument("--languages",  type=str, nargs="+",
                        default=["de", "fr", "es", "pt", "it"])
    parser.add_argument("--n-eval",     type=int, default=500,
                        help="Number of held-out sentence pairs to evaluate per language.")
    parser.add_argument("--max-seq-len", type=int, default=128)
    return parser.parse_args()


def load_student(checkpoint_path: Path, teacher_dim: int) -> StudentEncoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    student = StudentEncoder(teacher_dim=teacher_dim)
    student.load_state_dict(checkpoint["student_state_dict"])
    student.eval()
    return student


def embed_sentences(sentences, tokenizer, model, max_seq_len, batch_size=32):
    """Tokenize and embed a list of sentences in batches."""
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded = tokenizer(
                batch, max_length=max_seq_len, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            embeddings = model(encoded["input_ids"], encoded["attention_mask"])
            all_embeddings.append(embeddings)
    return torch.cat(all_embeddings, dim=0)


def main():
    args = parse_args()

    logger.info("Loading teacher and student...")
    teacher = TeacherEncoder()
    student = load_student(args.checkpoint, teacher.embedding_dim)
    teacher_tokenizer = teacher._st_model.tokenizer

    results = {}

    for lang in args.languages:
        logger.info(f"\nEvaluating {lang}...")
        english, translated = load_parallel_sentences(lang, args.n_eval, split="test")

        english_embeddings = embed_sentences(
            english, teacher_tokenizer, teacher, args.max_seq_len
        )
        translated_embeddings = embed_sentences(
            translated, student.tokenizer, student, args.max_seq_len
        )

        # Normalize teacher embeddings too, for fair cosine similarity comparison
        english_embeddings = torch.nn.functional.normalize(english_embeddings, p=2, dim=-1)

        correct_indices = list(range(len(english)))  # sentence i matches sentence i

        p1  = precision_at_k(translated_embeddings, english_embeddings, correct_indices, k=1)
        p5  = precision_at_k(translated_embeddings, english_embeddings, correct_indices, k=5)
        p10 = precision_at_k(translated_embeddings, english_embeddings, correct_indices, k=10)
        mrr = mean_reciprocal_rank(translated_embeddings, english_embeddings, correct_indices)

        results[lang] = {"P@1": p1, "P@5": p5, "P@10": p10, "MRR": mrr}
        logger.info(f"  P@1: {p1:.3f} | P@5: {p5:.3f} | P@10: {p10:.3f} | MRR: {mrr:.3f}")

    print("\n" + "=" * 60)
    print(f'{"Language":<10} {"P@1":>8} {"P@5":>8} {"P@10":>8} {"MRR":>8}')
    print("-" * 60)
    for lang, metrics in results.items():
        print(f'{lang:<10} {metrics["P@1"]:>8.3f} {metrics["P@5"]:>8.3f} '
              f'{metrics["P@10"]:>8.3f} {metrics["MRR"]:>8.3f}')
    print("-" * 60)
    avg_p1 = sum(m["P@1"] for m in results.values()) / len(results)
    avg_mrr = sum(m["MRR"] for m in results.values()) / len(results)
    print(f'{"AVERAGE":<10} {avg_p1:>8.3f} {"":>8} {"":>8} {avg_mrr:>8.3f}')
    print("=" * 60)


if __name__ == "__main__":
    main()
