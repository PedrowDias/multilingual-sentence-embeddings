import torch
from sentence_transformers import SentenceTransformer

from mse.models.base import SentenceEncoder


class TeacherEncoder(SentenceEncoder):
    '''Frozen English sentence encoder used as the distillation teacher.

    Wraps a pretrained SentenceTransformer model (default: all-MiniLM-L6-v2).
    The teacher is always frozen — its weights are never updated during
    distillation.

    Explicitly loads onto CPU: SentenceTransformer auto-detects available
    accelerators (e.g. Apple Silicon's MPS backend) and will silently place
    itself there if not told otherwise, causing device mismatches when the
    rest of the pipeline runs on CPU.

    Args:
        model_name: HuggingFace model name for the teacher encoder.
        device:     Device to load the model on. Defaults to 'cpu' to match
                    the rest of the training/evaluation pipeline.
    '''

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu') -> None:
        super().__init__()
        self._st_model = SentenceTransformer(model_name, device=device)
        self._embedding_dim = self._st_model.get_sentence_embedding_dimension()

        for param in self._st_model.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        output = self._st_model(features)
        return output['sentence_embedding']

    def encode_sentences(self, sentences: list[str]) -> torch.Tensor:
        embeddings = self._st_model.encode(
            sentences,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        return embeddings
