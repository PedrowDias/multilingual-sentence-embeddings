import torch
from torch.utils.data import DataLoader, TensorDataset

from mse.models.base import SentenceEncoder
from mse.training.config import DistillationConfig
from mse.training.distillation_trainer import DistillationTrainer


class MockEncoder(SentenceEncoder):
    '''A tiny mock encoder for fast, offline testing of the trainer.

    Maps token ids to embeddings via a simple learned linear layer, avoiding
    any dependency on downloading real pretrained models in tests.
    '''

    def __init__(self, vocab_size: int = 100, embed_dim: int = 8, trainable: bool = True):
        super().__init__()
        self._embed_dim = embed_dim
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self._embed_dim

    def forward(self, input_ids, attention_mask):
        # Mean pool over the embedding, ignoring padding
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (embedded * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-8)
        return summed / counts


def _make_loaders(n_samples=16, seq_len=8, vocab_size=100, batch_size=4):
    def random_batch():
        return {
            'teacher_input_ids': torch.randint(0, vocab_size, (n_samples, seq_len)),
            'teacher_attention_mask': torch.ones(n_samples, seq_len, dtype=torch.long),
            'student_input_ids': torch.randint(0, vocab_size, (n_samples, seq_len)),
            'student_attention_mask': torch.ones(n_samples, seq_len, dtype=torch.long),
        }

    data = random_batch()
    dataset = TensorDataset(
        data['teacher_input_ids'], data['teacher_attention_mask'],
        data['student_input_ids'], data['student_attention_mask'],
    )

    def collate(batch):
        teacher_ids, teacher_mask, student_ids, student_mask = zip(*batch)
        return {
            'teacher_input_ids': torch.stack(teacher_ids),
            'teacher_attention_mask': torch.stack(teacher_mask),
            'student_input_ids': torch.stack(student_ids),
            'student_attention_mask': torch.stack(student_mask),
        }

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    return loader, loader


class TestDistillationTrainer:

    def test_train_loss_recorded(self):
        teacher = MockEncoder(trainable=False)
        student = MockEncoder()
        train_loader, val_loader = _make_loaders()
        config = DistillationConfig(epochs=2, batch_size=4)
        trainer = DistillationTrainer(student, teacher, train_loader, val_loader, config)
        trainer.train()
        assert len(trainer.train_losses) == 2

    def test_val_loss_recorded(self):
        teacher = MockEncoder(trainable=False)
        student = MockEncoder()
        train_loader, val_loader = _make_loaders()
        config = DistillationConfig(epochs=2, batch_size=4)
        trainer = DistillationTrainer(student, teacher, train_loader, val_loader, config)
        trainer.train()
        assert len(trainer.val_losses) == 2

    def test_teacher_weights_never_change(self):
        teacher = MockEncoder(trainable=False)
        student = MockEncoder()
        initial_teacher_weights = teacher.embedding.weight.clone()

        train_loader, val_loader = _make_loaders()
        config = DistillationConfig(epochs=2, batch_size=4)
        trainer = DistillationTrainer(student, teacher, train_loader, val_loader, config)
        trainer.train()

        assert torch.equal(initial_teacher_weights, teacher.embedding.weight)

    def test_student_weights_do_change(self):
        teacher = MockEncoder(trainable=False)
        student = MockEncoder()
        initial_student_weights = student.embedding.weight.clone()

        train_loader, val_loader = _make_loaders()
        config = DistillationConfig(epochs=3, batch_size=4, learning_rate=1e-2)
        trainer = DistillationTrainer(student, teacher, train_loader, val_loader, config)
        trainer.train()

        assert not torch.equal(initial_student_weights, student.embedding.weight)

    def test_loss_decreases_on_easy_task(self):
        '''Student should learn to approximate a fixed frozen teacher over time.'''
        torch.manual_seed(0)
        teacher = MockEncoder(embed_dim=8, trainable=False)
        student = MockEncoder(embed_dim=8)

        train_loader, val_loader = _make_loaders(n_samples=8, batch_size=8)
        config = DistillationConfig(epochs=30, batch_size=8, learning_rate=1e-2)
        trainer = DistillationTrainer(student, teacher, train_loader, val_loader, config)
        trainer.train()

        assert trainer.train_losses[-1] < trainer.train_losses[0]
