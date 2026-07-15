from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

DEFAULT_CODEBOOK_DIR = Path(__file__).resolve().parent.parent / "codebooks"
BIT_SETTING_TO_M = {"2bit": 32, "1bit": 16}
KEY_HEADS_PER_GROUP = 2
VAL_HEADS_PER_GROUP = 8


def default_channel_groups(M: int) -> Tuple[List[int], List[int], List[int]]:
    if M == 32:
        high = [12, 13, 14, 15, 28, 29, 30, 31]
        mid = [4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24, 25, 26, 27]
        low = [0, 1, 2, 3, 16, 17, 18, 19]
    elif M == 16:
        high = [6, 7, 14, 15]
        mid = [2, 3, 4, 5, 10, 11, 12, 13]
        low = [0, 1, 8, 9]
    else:
        raise ValueError(f"Unsupported M={M}; expected 16 or 32")
    return high, mid, low


def resolve_packed_codebook_path(
    bit_setting: str,
    codebook_path: Optional[Union[str, Path]] = None,
    codebook_dir: Optional[Union[str, Path]] = None,
) -> Path:
    if bit_setting not in BIT_SETTING_TO_M:
        raise ValueError("bit_setting must be '1bit' or '2bit'")

    path = (
        Path(codebook_path)
        if codebook_path is not None
        else (Path(codebook_dir) if codebook_dir is not None else DEFAULT_CODEBOOK_DIR)
        / f"llama3_{bit_setting}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Packed codebook not found: {path}")
    return path


def _to_device(t: torch.Tensor, device) -> torch.Tensor:
    return t.to(device, non_blocking=True).contiguous()


def load_packed_codebooks(
    codebook_path: Union[str, Path],
    device,
    high_groups: Sequence[int],
    mid_groups: Sequence[int],
    low_groups: Sequence[int],
    num_layers: Optional[int] = None,
):
    path = Path(codebook_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != "meltkv_codebook_v1":
        raise ValueError(f"Unsupported codebook format in {path}")

    key, value = blob["key"], blob["value"]
    L_pack, Gk = key["high"].shape[:2]
    L = L_pack if num_layers is None else num_layers
    if L > L_pack:
        raise ValueError(f"Requested {L} layers but packed file has {L_pack}")

    for tier, n_expected in (
        ("high", len(high_groups)),
        ("mid", len(mid_groups)),
        ("low", len(low_groups)),
    ):
        if n_expected == 0:
            continue
        n_got = key[tier].shape[2]
        if n_got != n_expected:
            raise ValueError(
                f"Packed key[{tier}] has {n_got} channel groups, expected {n_expected}"
            )

    Gv = value.shape[1]
    k_high = [[] for _ in range(Gk)]
    k_mid = [[] for _ in range(Gk)]
    k_low = [[] for _ in range(Gk)]
    v_all = [[] for _ in range(Gv)]

    empty_mid = len(mid_groups) == 0
    for hg in range(Gk):
        for layer in range(L):
            k_high[hg].append(_to_device(key["high"][layer, hg], device))
            k_low[hg].append(_to_device(key["low"][layer, hg], device))
            if empty_mid:
                k_mid[hg].append(key["high"][layer, hg][:0].to(device).contiguous())
            else:
                k_mid[hg].append(_to_device(key["mid"][layer, hg], device))

    for hg in range(Gv):
        for layer in range(L):
            v_all[hg].append(_to_device(value[layer, hg], device))

    meta = {
        "path": str(path),
        "bit_setting": blob.get("bit_setting"),
        "M": blob.get("M"),
        "nbits": blob.get("nbits"),
        "num_layers": L,
        "key_heads_per_group": blob.get("key_heads_per_group"),
        "val_heads_per_group": blob.get("val_heads_per_group"),
    }
    return k_high, k_mid, k_low, v_all, meta
