import pytest
from transformers import AutoTokenizer

from mse.data.parallel_dataset import ParallelSentenceDataset


@pytest.fixture(scope='module')
def tiny_tokenizer():
    '''A small, fast tokenizer for testing — avoids downloading large models.'''
    return AutoTokenizer.from_pretrained('hf-internal-testing/tiny-random-bert')


class TestParallelSentenceDataset:

    def test_length(self, tiny_tokenizer):
        english = ['hello world', 'good morning']
        translated = ['ola mundo', 'bom dia']
        ds = ParallelSentenceDataset(english, translated, tiny_tokenizer, tiny_tokenizer, max_seq_len=16)
        assert len(ds) == 2

    def test_item_keys(self, tiny_tokenizer):
        english = ['hello world']
        translated = ['ola mundo']
        ds = ParallelSentenceDataset(english, translated, tiny_tokenizer, tiny_tokenizer, max_seq_len=16)
        item = ds[0]
        expected_keys = {
            'teacher_input_ids', 'teacher_attention_mask',
            'student_input_ids', 'student_attention_mask',
        }
        assert set(item.keys()) == expected_keys

    def test_item_shapes(self, tiny_tokenizer):
        english = ['hello world']
        translated = ['ola mundo']
        ds = ParallelSentenceDataset(english, translated, tiny_tokenizer, tiny_tokenizer, max_seq_len=16)
        item = ds[0]
        assert item['teacher_input_ids'].shape == (16,)
        assert item['student_input_ids'].shape == (16,)

    def test_mismatched_lengths_raises(self, tiny_tokenizer):
        with pytest.raises(ValueError, match='equal length'):
            ParallelSentenceDataset(
                ['a', 'b'], ['x'], tiny_tokenizer, tiny_tokenizer
            )
