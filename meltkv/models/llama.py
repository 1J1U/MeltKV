from typing import Optional, Tuple

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from transformers.models.llama.modeling_llama import (
    Cache,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)

from ..cache import MeltKVCache
from ..config import MeltKVConfig
from ..inject import ContextList, MethodInjector
from .hooks import ModelHooks


def _project_qkv(attn: LlamaSdpaAttention, hidden_states: torch.Tensor):
    bsz, q_len, _ = hidden_states.size()
    query_states = attn.q_proj(hidden_states)
    key_states = attn.k_proj(hidden_states)
    value_states = attn.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, attn.num_key_value_heads, attn.head_dim
    ).transpose(1, 2)
    return query_states, key_states, value_states, bsz, q_len


def _apply_rope(
    attn: LlamaSdpaAttention,
    query_states,
    key_states,
    value_states,
    position_ids,
    position_embeddings,
):
    if position_embeddings is None:
        cos, sin = attn.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    return query_states, key_states, cos, sin


def baseline_attention_forward(
    self: LlamaSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        return super(LlamaSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    query_states, key_states, value_states, bsz, q_len = _project_qkv(self, hidden_states)
    query_states, key_states, cos, sin = _apply_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

    attn_output = sdpa(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=causal_mask is None and q_len > 1,
    )
    attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
    return self.o_proj(attn_output), None, past_key_value


def meltkv_attention_forward(
    self: LlamaSdpaAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        logger.warning_once(
            "output_attentions=True is not supported with MeltKV; falling back to eager attention."
        )
        return super(LlamaSdpaAttention, self).forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

    query_states, key_states, value_states, bsz, q_len = _project_qkv(self, hidden_states)
    if q_len == 1:
        raise RuntimeError(
            "Token-by-token decoding is not supported in this PPL-only release. "
            "LongBench evaluation: coming soon."
        )

    assert MeltKVCache.has_instance(), "MeltKVCache must be created before evaluation."
    cache = MeltKVCache()
    distort_recent = bool(getattr(MeltKVConfig(), "distort_recent", True))

    key_states, value_states = cache.quantize(
        query_states,
        key_states,
        value_states,
        layer_idx=self.layer_idx,
        distort_recent=distort_recent,
    )
    query_states, key_states, _, _ = _apply_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )

    nrep = query_states.size(1) // key_states.size(1)
    key_states = repeat_kv(key_states, nrep)
    value_states = repeat_kv(value_states, nrep)
    attn_out = sdpa(query_states, key_states, value_states, is_causal=True)
    attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
    return self.o_proj(attn_out), None, past_key_value


model_hooks = ModelHooks(
    init_context=ContextList([]),
    baseline_context=ContextList(
        [MethodInjector(LlamaSdpaAttention, "forward", baseline_attention_forward)]
    ),
    evaluation_context=ContextList(
        [MethodInjector(LlamaSdpaAttention, "forward", meltkv_attention_forward)]
    ),
)
