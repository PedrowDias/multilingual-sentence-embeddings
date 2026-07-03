from dataclasses import dataclass


@dataclass
class DistillationConfig:
    '''Hyperparameters for the knowledge distillation training run.'''
    # Optimisation
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500

    # Learning rate schedule
    lr_decay_every: int = 1
    lr_decay_factor: float = 0.9

    # Data
    max_seq_len: int = 128
    languages: tuple[str, ...] = ('de', 'fr', 'es', 'pt', 'it')

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every_n_epochs: int = 1

    # Reproducibility
    seed: int = 42
    device: str = 'cpu'
