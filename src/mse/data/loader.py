import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_parallel_sentences(
    language: str,
    n_samples: int,
    split: str = 'train',
) -> tuple[list[str], list[str]]:
    '''Load parallel English-X sentence pairs from the OPUS-100 dataset.

    OPUS-100 config names are not consistently ordered — some pairs are named
    'en-X' and others 'X-en', depending on alphabetical ordering of the
    language codes (e.g. 'de-en' for German since 'd' < 'e', but 'en-fr' for
    French since 'e' < 'f'). This function tries both orderings automatically.

    Args:
        language:  Two-letter language code (e.g. 'fr', 'de', 'es', 'pt', 'it').
        n_samples: Number of sentence pairs to load.
        split:     Dataset split — 'train', 'validation', or 'test'.

    Returns:
        Tuple of (english_sentences, target_sentences), both length n_samples,
        aligned by index.

    Raises:
        ValueError: If the requested language pair is not available in either
                    ordering.
    '''
    candidate_configs = [f'en-{language}', f'{language}-en']

    dataset = None
    last_error = None
    for config_name in candidate_configs:
        try:
            dataset = load_dataset('Helsinki-NLP/opus-100', config_name, split=split)
            break
        except ValueError as e:
            last_error = e
            continue

    if dataset is None:
        raise ValueError(
            f"Could not load language pair for '{language}' from OPUS-100 "
            f'under either ordering {candidate_configs}. '
            f'Original error: {last_error}'
        )

    n_available = len(dataset)
    if n_samples > n_available:
        logger.warning(
            f'Requested {n_samples} samples but only {n_available} available '
            f'for {language} ({split} split). Using all {n_available}.'
        )
        n_samples = n_available

    subset = dataset.select(range(n_samples))

    english_sentences = [pair['translation']['en'] for pair in subset]
    target_sentences  = [pair['translation'][language] for pair in subset]

    logger.info(f'Loaded {len(english_sentences)} pairs for en-{language} ({split}).')
    return english_sentences, target_sentences
