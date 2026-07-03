import torch


def precision_at_k(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    correct_indices: list[int],
    k: int = 1,
) -> float:
    '''Compute Precision@k for cross-lingual retrieval.

    For each query, retrieves the top-k most similar corpus embeddings by
    cosine similarity, and checks whether the correct match is among them.

    Args:
        query_embeddings:  Shape (n_queries, dim). Embeddings of source
                           language sentences (e.g. Portuguese).
        corpus_embeddings: Shape (n_corpus, dim). Embeddings of target
                           language sentences (e.g. English).
        correct_indices:   correct_indices[i] is the index in corpus_embeddings
                           that is the correct match for query i.
        k:                 Number of top results to consider.

    Returns:
        Fraction of queries where the correct match appears in the top-k
        retrieved results. 1.0 = perfect retrieval.
    '''
    # Cosine similarity matrix — embeddings are L2-normalized so this is
    # equivalent to a dot product.
    scores = torch.mm(query_embeddings, corpus_embeddings.T)  # (n_queries, n_corpus)
    top_k  = scores.topk(k, dim=-1).indices                   # (n_queries, k)

    correct = torch.tensor(correct_indices)
    hits = (top_k == correct.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()


def mean_reciprocal_rank(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    correct_indices: list[int],
) -> float:
    '''Mean Reciprocal Rank (MRR) for cross-lingual retrieval.

    MRR is the average of 1/rank for each query, where rank is the position
    of the correct match in the ranked list. A higher value means correct
    matches appear closer to the top.

    Args:
        query_embeddings:  Shape (n_queries, dim).
        corpus_embeddings: Shape (n_corpus, dim).
        correct_indices:   Correct match index for each query.

    Returns:
        MRR score between 0 and 1. 1.0 = correct match always ranked first.
    '''
    scores  = torch.mm(query_embeddings, corpus_embeddings.T)
    # argsort descending: position 0 = most similar
    ranking = scores.argsort(dim=-1, descending=True)

    reciprocal_ranks = []
    for i, correct_idx in enumerate(correct_indices):
        rank = (ranking[i] == correct_idx).nonzero(as_tuple=True)[0].item()
        reciprocal_ranks.append(1.0 / (rank + 1))

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
