from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from pykeops.torch import LazyTensor

from .singleton import Singleton


class MeltKVCache(metaclass=Singleton):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def has_instance(cls):
        return cls._instance is not None

    def __init__(
        self,
        *,
        bs,
        nh,
        num_key_value_heads,
        M,
        layer_num,
        dtype=torch.uint8,
        nbits=8,
        d=128,
        scalar_t=torch.float32,
        device=None,
    ):
        self.bs = bs
        self.nh = nh
        self.num_key_value_heads = num_key_value_heads
        self.M = M
        self.layer_num = layer_num
        self.dtype = dtype
        self.nbits = nbits
        self.d = d
        self.scalar_t = scalar_t
        self.device = device or "cuda"

        self.key_heads_per_group = 2
        self.val_heads_per_group = 8
        self.heads_per_group = 2

        self.k_high = None
        self.k_mid = None
        self.k_low = None
        self.v_all = None
        self.high_groups: List[int] = []
        self.mid_groups: List[int] = []
        self.low_groups: List[int] = []

        self.init_cache()

    def init_cache(self):
        self.key_cache_highlow = [None for _ in range(self.layer_num)]
        self.key_cache_mid = [None for _ in range(self.layer_num)]
        self.value_cache = [None for _ in range(self.layer_num)]
        self.seen_tokens = [0 for _ in range(self.layer_num)]

    def set_tiered_codebooks(
        self,
        k_high,
        k_mid,
        k_low,
        v_all,
        heads_per_group,
        M,
        high_groups: Optional[Sequence[int]] = None,
        mid_groups: Optional[Sequence[int]] = None,
        low_groups: Optional[Sequence[int]] = None,
        key_heads_per_group=None,
        val_heads_per_group=None,
        **kwargs,
    ):
        if high_groups is None or mid_groups is None or low_groups is None:
            raise ValueError("high_groups, mid_groups, and low_groups are required.")

        self.k_high = k_high
        self.k_mid = k_mid
        self.k_low = k_low
        self.v_all = v_all
        self.M = M
        self.key_heads_per_group = int(
            key_heads_per_group if key_heads_per_group is not None else heads_per_group
        )
        self.val_heads_per_group = int(
            val_heads_per_group if val_heads_per_group is not None else 8
        )
        self.heads_per_group = self.key_heads_per_group
        self.high_groups = list(high_groups)
        self.mid_groups = list(mid_groups)
        self.low_groups = list(low_groups)

    @staticmethod
    def encode_tier_subset(X, C_full, groups, *, target_dtype=torch.uint8):
        B, H, N, D = X.shape
        idx = torch.tensor(groups, device=X.device, dtype=torch.long)
        G = idx.numel()
        if G == 0:
            return torch.zeros(B, H, N, 0, device=X.device, dtype=target_dtype)

        d_m = C_full.size(-1)
        M_total = D // d_m
        Xg = X.view(B, H, N, M_total, d_m).index_select(3, idx)

        if C_full.dim() == 3:
            Cg = C_full if C_full.size(0) == G else C_full.index_select(0, idx)
            Cg = Cg.unsqueeze(0).expand(H, -1, -1, -1)
        else:
            Cg = C_full if C_full.size(1) == G else C_full.index_select(1, idx)

        Xflat = Xg.permute(0, 1, 3, 2, 4).reshape(B * H * G, N, d_m).contiguous().float()
        Cflat = Cg.reshape(H * G, -1, d_m).repeat(B, 1, 1).contiguous().float()

        x_i = LazyTensor(Xflat[:, :, None, :])
        c_j = LazyTensor(Cflat[:, None, :, :])
        indices = ((x_i - c_j) ** 2).sum(-1).argmin(dim=2)
        return indices.view(B, H, G, N).permute(0, 1, 3, 2).contiguous().to(target_dtype)

    @staticmethod
    def decode_tier_subset(codes_g, C_full, groups, out_dtype=None, M_total=None, D_total=None):
        B, H, N, G = codes_g.shape
        device = codes_g.device
        idx = torch.tensor(groups, device=device, dtype=torch.long)
        d_m = C_full.size(-1)

        if idx.numel() == 0:
            if M_total is None:
                M_total = 0 if D_total is None else D_total // d_m
            dtype = out_dtype or torch.float16
            return torch.zeros(B, H, N, M_total * d_m, device=device, dtype=dtype)

        if D_total is not None:
            M_total = D_total // d_m
        elif M_total is None:
            M_total = int(idx.max().item()) + 1

        if C_full.dim() == 3:
            Cg = C_full if C_full.size(0) == G else C_full.index_select(0, idx)
            Cg = Cg.unsqueeze(0).expand(H, -1, -1, -1)
        else:
            Cg = C_full if C_full.size(1) == G else C_full.index_select(1, idx)

        codes = codes_g.long().unsqueeze(-1).expand(-1, -1, -1, -1, d_m)
        C_exp = Cg.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1, -1)
        gathered = torch.gather(C_exp, 4, codes.unsqueeze(-2)).squeeze(4)

        dtype = gathered.dtype if out_dtype is None else out_dtype
        Xfull = torch.zeros(B, H, N, M_total, d_m, device=device, dtype=dtype)
        Xfull.index_copy_(3, idx, gathered.to(dtype))
        return Xfull.view(B, H, N, M_total * d_m)

    def store_prefill_codes(self, layer_idx, key_codes_highlow, key_codes_mid, value_codes):
        self.key_cache_highlow[layer_idx] = key_codes_highlow
        self.key_cache_mid[layer_idx] = key_codes_mid
        self.value_cache[layer_idx] = value_codes
        self.seen_tokens[layer_idx] = key_codes_highlow.size(2)

    def dequantize_cached(self, query_states, key_states, value_states, layer_idx):
        key_codes_highlow = self.key_cache_highlow[layer_idx]
        key_codes_mid = self.key_cache_mid[layer_idx]
        value_codes = self.value_cache[layer_idx]
        if key_codes_highlow is None or key_codes_mid is None:
            return key_states, value_states

        B, H, Tk, _ = key_codes_highlow.shape
        key_hpg = self.key_heads_per_group
        val_hpg = self.val_heads_per_group
        num_groups_k = H // key_hpg
        num_groups_v = H // val_hpg

        key_codes_high = (key_codes_highlow.to(torch.int32) >> 6) & 0x3FF
        key_codes_low = key_codes_highlow.to(torch.int32) & 0x3F

        key_decoded_groups = []
        for hg in range(num_groups_k):
            h0, h1 = hg * key_hpg, (hg + 1) * key_hpg
            dec_high = self.decode_tier_subset(
                key_codes_high[:, h0:h1],
                self.k_high[hg][layer_idx],
                self.high_groups,
                out_dtype=torch.float16,
                M_total=self.M,
            )
            dec_mid = self.decode_tier_subset(
                key_codes_mid[:, h0:h1],
                self.k_mid[hg][layer_idx],
                self.mid_groups,
                out_dtype=torch.float16,
                M_total=self.M,
            )
            dec_low = self.decode_tier_subset(
                key_codes_low[:, h0:h1],
                self.k_low[hg][layer_idx],
                self.low_groups,
                out_dtype=torch.float16,
                M_total=self.M,
            )
            key_decoded_groups.append(dec_high + dec_mid + dec_low)
        key_out = torch.cat(key_decoded_groups, dim=1)

        value_decoded_groups = []
        for hg in range(num_groups_v):
            h0, h1 = hg * val_hpg, (hg + 1) * val_hpg
            value_decoded_groups.append(
                pq_decode(value_codes[:, h0:h1], self.v_all[hg][layer_idx]).to(torch.float16)
            )
        value_out = torch.cat(value_decoded_groups, dim=1)
        return key_out, value_out

    def quantize(
        self,
        query_states,
        key_states,
        value_states,
        layer_idx,
        distort_recent: bool = True,
        tau=None,
    ):
        B, H, Tk, D = key_states.shape
        assert D % self.M == 0, f"D({D}) must be multiple of M({self.M})"
        device = key_states.device

        topk = 1
        key_fp16 = key_states[:, :, :topk, :]
        value_fp16 = value_states[:, :, :topk, :]
        key_quant = key_states[:, :, topk:, :]
        value_quant = value_states[:, :, topk:, :]

        key_hpg = self.key_heads_per_group
        val_hpg = self.val_heads_per_group
        num_groups_k = H // key_hpg
        num_groups_v = H // val_hpg

        key_codes_highlow_list = []
        key_codes_mid_list = []
        for hg in range(num_groups_k):
            h0, h1 = hg * key_hpg, (hg + 1) * key_hpg
            key_g = key_quant[:, h0:h1]
            key_codes_high = self.encode_tier_subset(
                key_g, self.k_high[hg][layer_idx], self.high_groups, target_dtype=torch.uint16
            )
            key_codes_mid = self.encode_tier_subset(
                key_g, self.k_mid[hg][layer_idx], self.mid_groups, target_dtype=torch.uint8
            )
            key_codes_low = self.encode_tier_subset(
                key_g, self.k_low[hg][layer_idx], self.low_groups, target_dtype=torch.uint8
            )
            key_codes_highlow = (
                (key_codes_high.to(torch.int32) << 6) | (key_codes_low.to(torch.int32) & 0x3F)
            ).to(torch.uint16)
            key_codes_highlow_list.append(key_codes_highlow)
            key_codes_mid_list.append(key_codes_mid)

        value_codes_list = []
        for hg in range(num_groups_v):
            h0, h1 = hg * val_hpg, (hg + 1) * val_hpg
            value_codes_list.append(
                pq_encode(value_quant[:, h0:h1], self.v_all[hg][layer_idx], target_dtype=self.dtype)
            )

        self.store_prefill_codes(
            layer_idx,
            torch.cat(key_codes_highlow_list, dim=1),
            torch.cat(key_codes_mid_list, dim=1),
            torch.cat(value_codes_list, dim=1),
        )

        if not distort_recent:
            key_out = torch.empty_like(key_states)
            value_out = torch.empty_like(value_states)
            key_out[:, :, :topk] = key_fp16
            key_out[:, :, topk:] = key_quant
            value_out[:, :, :topk] = value_fp16
            value_out[:, :, topk:] = value_quant
            return key_out, value_out

        key_quant_dec, value_quant_dec = self.dequantize_cached(
            query_states, key_quant, value_quant, layer_idx
        )
        if key_quant_dec.dim() == 3:
            key_quant_dec = key_quant_dec.unsqueeze(0)
        if value_quant_dec.dim() == 3:
            value_quant_dec = value_quant_dec.unsqueeze(0)

        key_out = torch.empty_like(key_states)
        value_out = torch.empty_like(value_states)
        key_out[:, :, :topk] = key_fp16
        key_out[:, :, topk:] = key_quant_dec[:, :, : Tk - topk]
        value_out[:, :, :topk] = value_fp16
        value_out[:, :, topk:] = value_quant_dec[:, :, : Tk - topk]
        return key_out, value_out


def pq_encode(X, C, target_dtype=torch.uint8):
    bs, num_heads, n, d = X.shape
    if C.dim() == 3:
        M, c, d_m = C.shape
        C = C.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, -1, -1, -1)
    else:
        bs, num_heads, M, c, d_m = C.shape

    d_m = d // M
    X = X.reshape(bs, num_heads, n, M, d_m)
    X = X.permute(0, 1, 3, 2, 4).reshape(bs * num_heads * M, n, d_m).contiguous().float()
    C = C.reshape(bs * num_heads * M, -1, d_m).contiguous().float()

    x_i = LazyTensor(X[:, :, None, :])
    c_j = LazyTensor(C[:, None, :, :])
    indices = ((x_i - c_j) ** 2).sum(-1).argmin(dim=2)
    return indices.view(bs, num_heads, M, n).permute(0, 1, 3, 2).contiguous().to(target_dtype)


def pq_decode(codes, C):
    bs, num_heads, n, M = codes.shape
    if C.dim() == 3:
        M, c, d_m = C.shape
        C = C.unsqueeze(0).unsqueeze(0).expand(bs, num_heads, -1, -1, -1)
    else:
        bs, num_heads, M, c, d_m = C.shape

    codes = codes.to(torch.long, non_blocking=True).unsqueeze(-1).expand(-1, -1, -1, -1, d_m)
    C_expanded = C.unsqueeze(2).expand(-1, -1, n, -1, -1, -1)
    decoded = torch.gather(C_expanded, 4, codes.unsqueeze(-2)).squeeze(4)
    return decoded.reshape(bs, num_heads, n, M * d_m)


def nbits_to_dtype(nbits):
    if nbits <= 8:
        return torch.uint8
    if nbits <= 16:
        return torch.uint16
    if nbits <= 32:
        return torch.uint32
    if nbits <= 64:
        return torch.uint64
    raise ValueError("nbits must be <= 64")
