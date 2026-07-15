PPL_DATASETS = [
    "wikitext-2-raw-v1",
    "wikitext-103-v1",
    "ptb_text_only",
    "wikitext-103-raw-v1",
]


def get_evaluator(dataset: str):
    if dataset in PPL_DATASETS:
        from .perplexity import run_perplexity
        return run_perplexity
    raise ValueError(
        f"Unknown dataset '{dataset}'. Supported PPL: {PPL_DATASETS}. "
        f"LongBench evaluation: coming soon."
    )
