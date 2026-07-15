#!/usr/bin/env python3


from __future__ import annotations

import argparse
import pathlib
import random
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from meltkv.cache import MeltKVCache, nbits_to_dtype
from meltkv.codebooks import (
    BIT_SETTING_TO_M,
    KEY_HEADS_PER_GROUP,
    VAL_HEADS_PER_GROUP,
    default_channel_groups,
    load_packed_codebooks,
    resolve_packed_codebook_path,
)
from meltkv.config import MeltKVConfig, load_config
from meltkv.eval import get_evaluator
from meltkv.models import get_model_hooks
from meltkv.timer import Timer, tprint


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    assert torch.cuda.is_available(), "CUDA is required"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="MeltKV perplexity evaluation")
    p.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Config json under configs/ (e.g. llama-3.1-8b.json)",
    )
    p.add_argument("-d", "--dataset", type=str, required=True, help="PPL dataset name")
    p.add_argument(
        "-M",
        type=int,
        default=None,
        help="Number of PQ subspaces (must match bit-setting: 2bit→32, 1bit→16)",
    )
    p.add_argument("--nbits", type=int, default=None, help="Bits per subspace (value cache dtype)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--half", action="store_true", help="Use float16 weights")
    p.add_argument("--breakdown", action="store_true")
    p.add_argument(
        "-p",
        "--pipeline",
        nargs="+",
        choices=["baseline", "evaluation"],
        default=["evaluation"],
    )
    p.add_argument(
        "--bit-setting",
        choices=["1bit", "2bit"],
        default="2bit",
        help="Select packed codebook: codebooks/llama3_{1bit,2bit}.pt",
    )
    p.add_argument(
        "--codebook",
        type=str,
        default=None,
        help="Path to a packed codebook .pt (default: codebooks/llama3_{bit}.pt)",
    )
    p.add_argument("--model-path", type=str, default=None, help="Override HF model path")
    p.add_argument("--datasets-root", type=str, default=None, help="Override datasets root")
    p.add_argument(
        "--no-distort-recent",
        action="store_true",
        help="Do not reconstruct quantized KV during prefill (faster, lower fidelity)",
    )
    return p.parse_args()


def resolve_M(bit_setting: str, M: int | None) -> int:

    expected = BIT_SETTING_TO_M[bit_setting]
    if M is None:
        return expected
    if int(M) != expected:
        raise ValueError(
            f"-M {M} is incompatible with --bit-setting {bit_setting}. "
            f"Use -M {expected} (2bit→32, 1bit→16)."
        )
    return expected


def main():
    args = parse_args()
    Timer("").start()

    config = MeltKVConfig()
    config.device = "cuda"
    config.root = ROOT
    config.config_root = ROOT / "configs"
    config += load_config(config.config_root / "default.json")
    config += load_config(config.config_root / args.file)

    config.bit_setting = args.bit_setting

    config.M = resolve_M(args.bit_setting, args.M)
    if args.nbits is not None:
        config.nbits = args.nbits
    config.dataset = args.dataset
    config.pipeline = args.pipeline
    config.seed = args.seed
    config.half = args.half
    config.breakdown = args.breakdown
    config.distort_recent = not args.no_distort_recent
    config.quant_rope = "pre"
    config.scalar_t = torch.float16 if config.half else torch.float32
    config.cache_dtype = nbits_to_dtype(config.nbits)

    if args.model_path:
        config.model_path = pathlib.Path(args.model_path)
    else:
        folder = pathlib.Path(config.folder)
        config.model_path = folder if folder.is_absolute() else (ROOT / "models" / folder)

    if args.datasets_root:
        config.datasets_root = pathlib.Path(args.datasets_root)
    else:
        config.datasets_root = ROOT / "datasets"

    seed_everything(config.seed)

    config.model_config = AutoConfig.from_pretrained(config.model_path)
    if config.model_config.model_type != "llama":
        raise ValueError("This release supports LLaMA family models only.")
    config.context = get_model_hooks(config.model_config.model_type)

    key_hpg, val_hpg = KEY_HEADS_PER_GROUP, VAL_HEADS_PER_GROUP
    config.key_heads_per_group = key_hpg
    config.val_heads_per_group = val_hpg

    tprint(f"Loading model from {config.model_path}")
    tprint(
        f"bit_setting={config.bit_setting} M={config.M} "
        f"key_hpg={key_hpg} val_hpg={val_hpg} quant_rope={config.quant_rope}"
    )
    with config.context.init_context:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            attn_implementation="sdpa",
        ).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if config.half:
            model = model.half()


    if "baseline" in config.pipeline:
        tprint("Baseline")
        evaluator = get_evaluator(config.dataset)
        with config.context.baseline_context:
            score = evaluator(model, tokenizer, **config.to_dict())
        tprint(f"Baseline score: {score}")


    if "evaluation" not in config.pipeline:
        tprint("Exit")
        return

    L = config.model_config.num_hidden_layers
    H_kv = config.model_config.num_key_value_heads
    device = config.device
    high_groups, mid_groups, low_groups = default_channel_groups(config.M)

    codebook_path = resolve_packed_codebook_path(
        config.bit_setting,
        codebook_path=args.codebook,
        codebook_dir=ROOT / "codebooks",
    )
    tprint(f"Loading packed codebooks: {codebook_path}")
    k_high, k_mid, k_low, v_all, cb_meta = load_packed_codebooks(
        codebook_path,
        device,
        high_groups,
        mid_groups,
        low_groups,
        num_layers=L,
    )
    if cb_meta.get("M") is not None and int(cb_meta["M"]) != int(config.M):
        raise ValueError(
            f"Config M={config.M} does not match packed codebook M={cb_meta['M']}. "
            f"Use --bit-setting 2bit with -M 32, or 1bit with -M 16."
        )
    meta_key_hpg = int(cb_meta.get("key_heads_per_group") or key_hpg)
    meta_val_hpg = int(cb_meta.get("val_heads_per_group") or val_hpg)
    if meta_key_hpg != key_hpg or meta_val_hpg != val_hpg:
        raise ValueError(
            f"Packed codebook head-groups key/val={meta_key_hpg}/{meta_val_hpg} "
            f"(expected {key_hpg}/{val_hpg})."
        )
    if H_kv % key_hpg != 0 or H_kv % val_hpg != 0:
        raise ValueError(
            f"num_key_value_heads={H_kv} must be divisible by "
            f"key_hpg={key_hpg} and val_hpg={val_hpg}."
        )

    cache = MeltKVCache(
        bs=1,
        num_key_value_heads=H_kv,
        nh=config.model_config.num_attention_heads,
        M=config.M,
        layer_num=L,
        dtype=config.cache_dtype,
        nbits=config.nbits,
        d=config.d,
        scalar_t=config.scalar_t,
        device=device,
    )
    cache.set_tiered_codebooks(
        k_high,
        k_mid,
        k_low,
        v_all,
        heads_per_group=key_hpg,
        M=config.M,
        high_groups=high_groups,
        mid_groups=mid_groups,
        low_groups=low_groups,
        key_heads_per_group=key_hpg,
        val_heads_per_group=val_hpg,
    )

    evaluator = get_evaluator(config.dataset)
    torch.cuda.empty_cache()

    with config.context.evaluation_context:
        score = evaluator(
            model,
            tokenizer,
            cache_clear_func=cache.init_cache,
            **config.to_dict(),
        )

    tprint(f"Score: {score}")
    tprint("Exit")


if __name__ == "__main__":
    main()
