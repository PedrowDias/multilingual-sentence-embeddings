import torch
import pytest
from mse.evaluation.metrics import precision_at_k, mean_reciprocal_rank


class TestPrecisionAtK:

    def test_perfect_retrieval(self):
        # query i matches corpus i exactly
        embeddings = torch.eye(5)
        correct = list(range(5))
        assert precision_at_k(embeddings, embeddings, correct, k=1) == 1.0

    def test_zero_retrieval(self):
        # queries match nothing in corpus
        queries = torch.eye(3)
        corpus  = -torch.eye(3)  # opposite directions
        correct = [0, 1, 2]
        assert precision_at_k(queries, corpus, correct, k=1) == 0.0

    def test_top3_catches_second_best(self):
        # correct match is at rank 2 — should be caught by k=3 but not k=1
        queries = torch.tensor([[1.0, 0.0]])
        corpus  = torch.tensor([
            [0.0, 1.0],   # rank 2 (less similar)
            [1.0, 0.0],   # rank 1 (most similar) — but wrong
            [0.9, 0.1],   # rank 3
        ])
        # correct is index 0
        assert precision_at_k(queries, corpus, [0], k=1) == 0.0
        assert precision_at_k(queries, corpus, [0], k=3) == 1.0

    def test_output_is_float(self):
        embeddings = torch.randn(4, 8)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        result = precision_at_k(embeddings, embeddings, list(range(4)), k=1)
        assert isinstance(result, float)


class TestMeanReciprocalRank:

    def test_perfect_mrr(self):
        embeddings = torch.eye(4)
        assert mean_reciprocal_rank(embeddings, embeddings, list(range(4))) == 1.0

    def test_mrr_is_between_zero_and_one(self):
        embeddings = torch.randn(5, 8)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        mrr = mean_reciprocal_rank(embeddings, embeddings, list(range(5)))
        assert 0.0 <= mrr <= 1.0
