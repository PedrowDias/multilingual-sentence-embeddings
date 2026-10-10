import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ParallelSentenceDataset(Dataset):
    '''Dataset of (English sentence, translated sentence) pairs.

    Used for knowledge distillation: the English side is fed to the frozen
    teacher, the translated side is fed to the student being trained. Both
    are tokenized on the fly using their respective tokenizers.

    Args:
        english_sentences:    List of English sentences.
        translated_sentences: List of translated sentences, same length and
                              order as english_sentences — translated_sentences[i]
                              is the translation of english_sentences[i].
        teacher_tokenizer:    Tokenizer for the teacher (English) model.
        student_tokenizer:    Tokenizer for the student (multilingual) model.
        max_seq_len:          Maximum token sequence length. Longer sentences
                              are truncated, shorter ones padded.
    '''

    def __init__(
        self,
        english_sentences: list[str],
        translated_sentences: list[str],
        teacher_tokenizer: PreTrainedTokenizer,
        student_tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 128,
    ) -> None:
        if len(english_sentences) != len(translated_sentences):
            raise ValueError(
                f'english_sentences and translated_sentences must have equal '
                f'length, got {len(english_sentences)} and {len(translated_sentences)}.'
            )

        self.english_sentences = english_sentences
        self.translated_sentences = translated_sentences
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.english_sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        english = self.english_sentences[idx]
        translated = self.translated_sentences[idx]

        teacher_enc = self.teacher_tokenizer(
            english,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        student_enc = self.student_tokenizer(
            translated,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'teacher_input_ids': teacher_enc['input_ids'].squeeze(0),
            'teacher_attention_mask': teacher_enc['attention_mask'].squeeze(0),
            'student_input_ids': student_enc['input_ids'].squeeze(0),
            'student_attention_mask': student_enc['attention_mask'].squeeze(0),
        }
