from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SentenceEncoder(ABC, nn.Module):
    '''Abstract interface for all sentence encoder models.

    A sentence encoder maps a batch of tokenized sentences to a batch of
    fixed-size embedding vectors. The embedding dimension is determined by
    the concrete implementation.

    Both the teacher and student models satisfy this interface, which lets
    the distillation trainer depend only on the abstraction rather than any
    specific model architecture.
    '''

    @abstractmethod
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''Encode a batch of tokenized sentences into embedding vectors.

        Args:
            input_ids:      Token ids, shape (batch, seq_len).
            attention_mask: Mask for padding tokens, shape (batch, seq_len).
                            1 for real tokens, 0 for padding.

        Returns:
            Embedding tensor, shape (batch, embedding_dim).
        '''

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        '''Dimensionality of the output embedding vectors.'''

    def parameter_count(self) -> int:
        '''Total number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
