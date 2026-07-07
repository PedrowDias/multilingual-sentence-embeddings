import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mse.models.student import StudentEncoder
from mse.models.teacher import TeacherEncoder
from mse.training.config import DistillationConfig

logger = logging.getLogger(__name__)


class DistillationTrainer:
    '''Trains a multilingual student encoder to match a frozen English teacher.

    Args:
        student:      The multilingual encoder being trained.
        teacher:      The frozen English encoder providing target embeddings.
        train_loader: DataLoader yielding batches with teacher_input_ids,
                      teacher_attention_mask, student_input_ids,
                      student_attention_mask.
        val_loader:   Same structure, held out for validation.
        config:       DistillationConfig with hyperparameters.
    '''

    def __init__(
        self,
        student: StudentEncoder,
        teacher: TeacherEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DistillationConfig,
    ) -> None:
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.teacher.eval()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.lr_decay_every,
            gamma=config.lr_decay_factor,
        )

        self.loss_fn = nn.MSELoss()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._validate_epoch()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(
                f'Epoch {epoch}/{self.config.epochs} — '
                f'train loss: {train_loss:.5f}, val loss: {val_loss:.5f}'
            )

            self.scheduler.step()

            if epoch % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, epoch: int) -> float:
        self.student.train()
        total_loss = 0.0

        progress = tqdm(
            self.train_loader,
            desc=f'Epoch {epoch}/{self.config.epochs} [train]',
            leave=False,
        )
        for batch in progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                teacher_emb = self.teacher(
                    batch['teacher_input_ids'], batch['teacher_attention_mask']
                )

            self.optimizer.zero_grad()
            student_emb = self.student(
                batch['student_input_ids'], batch['student_attention_mask']
            )
            loss = self.loss_fn(student_emb, teacher_emb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f'{loss.item():.5f}')

        return total_loss / len(self.train_loader)

    def _validate_epoch(self) -> float:
        self.student.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating', leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                teacher_emb = self.teacher(
                    batch['teacher_input_ids'], batch['teacher_attention_mask']
                )
                student_emb = self.student(
                    batch['student_input_ids'], batch['student_attention_mask']
                )
                loss = self.loss_fn(student_emb, teacher_emb)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        logger.info(f'Checkpoint saved to {path}.')
