import torch
from sentence_transformers import SentenceTransformer

from mse.models.base import SentenceEncoder


class TeacherEncoder(SentenceEncoder):
    '''Frozen English sentence encoder used as the distillation teacher.

    Wraps a pretrained SentenceTransformer model (default: all-MiniLM-L6-v2,
    a strong, fast English encoder). The teacher is always frozen — its weights
    are never updated during distillation. Its role is solely to provide target
    embeddings for English sentences, which the student learns to match for
    sentences in other languages.

    Args:
        model_name: HuggingFace model name for the teacher encoder.
                    Must be a sentence-transformers compatible model.
    '''

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        super().__init__()
        self._st_model = SentenceTransformer(model_name)
        self._embedding_dim = self._st_model.get_sentence_embedding_dimension()

        # Freeze all parameters — teacher is never updated
        for param in self._st_model.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        '''Encode a batch of tokenized English sentences.

        Args:
            input_ids:      Shape (batch, seq_len).
            attention_mask: Shape (batch, seq_len).

        Returns:
            Normalized embeddings, shape (batch, embedding_dim).
        '''
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        output = self._st_model(features)
        return output['sentence_embedding']

    def encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        '''Convenience method: encode raw strings directly without tokenizing.

        Used during evaluation when we want to embed full sentences without
        going through the data pipeline.

        Args:
            sentences: List of English strings.

        Returns:
            Embeddings, shape (len(sentences), embedding_dim).
        '''
        embeddings = self._st_model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings
