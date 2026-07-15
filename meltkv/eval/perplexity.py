import torch
from datasets import load_from_disk
from tqdm import tqdm

from ..timer import tprint


def _load_dataset(dataset_name, root):
    supported = [
        "wikitext-2-raw-v1",
        "wikitext-103-v1",
        "ptb_text_only",
        "wikitext-103-raw-v1",
    ]
    if dataset_name not in supported:
        raise ValueError(f"Dataset {dataset_name} not supported for perplexity")

    dataset = load_from_disk(str(root / dataset_name))["test"]
    if dataset_name == "ptb_text_only":
        dataset = dataset.rename_column("sentence", "text")
    return dataset


def _encode_dataset(dataset, tokenizer):
    merged_text = "\n\n".join(dataset["text"])
    return tokenizer(merged_text, return_tensors="pt")


def run_perplexity(
    model,
    tokenizer,
    *,
    dataset,
    datasets_root,
    device,
    max_length,
    stride=None,
    verbose=True,
    cache_clear_func=None,
    **kwargs,
):
    model.to(device)
    model.eval()

    if verbose:
        tprint(f"Loading dataset {dataset}")
    raw = _load_dataset(dataset, datasets_root)
    if verbose:
        tprint("Encoding dataset")
    encodings = _encode_dataset(raw, tokenizer)

    stride = stride or max_length
    seq_len = encodings["input_ids"].size(1)

    nlls = []
    total_length = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len != max_length:
            continue

        input_ids = encodings["input_ids"][:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if cache_clear_func is not None:
            cache_clear_func()

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll = outputs.loss

        nlls.append(nll * trg_len)
        total_length += trg_len
        if verbose:
            print(f"current ppl: {torch.exp(torch.stack(nlls).sum() / total_length)}")

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_length)
    torch.cuda.empty_cache()
    return ppl.item()
