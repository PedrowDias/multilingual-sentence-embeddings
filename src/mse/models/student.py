import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from mse.models.base import SentenceEncoder


class StudentEncoder(SentenceEncoder):
    '''Multilingual sentence encoder trained via knowledge distillation.

    Wraps a pretrained multilingual transformer (default: XLM-RoBERTa-base)
    and adds a mean pooling layer to produce fixed-size sentence embeddings.
    Unlike the teacher, the student's weights are updated during distillation.

    The student is initialized from a multilingual pretrained model, which
    already has some cross-lingual alignment from its pretraining. Distillation
    fine-tunes this alignment so that the student's embeddings in any language
    match the teacher's English embeddings for the same meaning.

    Mean pooling: average the token embeddings weighted by the attention mask,
    ignoring padding tokens. This is the standard pooling strategy for
    sentence-transformers and outperforms CLS token pooling on most tasks.

    Args:
        model_name:     HuggingFace model name for the multilingual base.
        projection_dim: If provided, adds a linear projection to map from the
                        model's hidden size to this dimension, so the student's
                        embedding dim matches the teacher's. If None, uses the
                        model's native hidden size.
        teacher_dim:    Embedding dimension of the teacher. Used to set up the
                        projection layer if projection_dim is None.
    '''

    def __init__(
        self,
        model_name: str = 'xlm-roberta-base',
        teacher_dim: int = 384,
    ) -> None:
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._encoder   = AutoModel.from_pretrained(model_name)
        hidden_size     = self._encoder.config.hidden_size

        # Project from model hidden size (768 for XLM-R-base) to teacher dim (384)
        # so that MSE loss is computed in a comparable embedding space.
        self._projection = nn.Linear(hidden_size, teacher_dim)
        self._teacher_dim = teacher_dim

    @property
    def embedding_dim(self) -> int:
        return self._teacher_dim

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''Encode a batch of tokenized multilingual sentences.

        Args:
            input_ids:      Shape (batch, seq_len).
            attention_mask: Shape (batch, seq_len).

        Returns:
            Projected, normalized embeddings, shape (batch, teacher_dim).
        '''
        outputs = self._encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self._mean_pool(outputs.last_hidden_state, attention_mask)
        projected = self._projection(pooled)
        # L2-normalize so cosine similarity == dot product — standard practice
        return nn.functional.normalize(projected, p=2, dim=-1)

    def _mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        '''Average token embeddings, ignoring padding.

        Args:
            token_embeddings: Shape (batch, seq_len, hidden_size).
            attention_mask:   Shape (batch, seq_len).

        Returns:
            Shape (batch, hidden_size).
        '''
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-8)
        return summed / counts
