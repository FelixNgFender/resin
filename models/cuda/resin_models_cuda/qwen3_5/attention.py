"""Qwen3.5 FlashInfer paged-attention kernels."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from resin.backend.cuda.paged_kv import SlabPagedKvCache
from resin_models_cuda.paged_kv import append_slab_kv

append_kv = append_slab_kv

# FlashInfer always splits KV under CUDA graphs when allowed, with a fixed grid
# sized to the device. Without it a batch-1 decode launches one CTA per KV head
# (4 of 188 SMs on an RTX PRO 6000), so long-context decode attention becomes
# latency bound. Splitting changes only the softmax merge order.
GRAPH_SPLIT_KV = os.environ.get("RESIN_QWEN_GRAPH_SPLIT_KV", "1") != "0"


def _plan_wrapper(
    wrapper,
    tokens: int,
    kv_indptr,
    pages,
    last_page_lengths,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    page_tokens: int,
    common: dict[str, Any],
    causal: bool = True,
) -> None:
    args = (
        kv_indptr,
        pages,
        last_page_lengths,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        page_tokens,
    )
    if tokens == 1:
        wrapper.plan(*args, **common)
    else:
        qo_indptr = (
            torch.arange(len(kv_indptr), dtype=torch.int32, device=kv_indptr.device)
            * tokens
        )
        wrapper.plan(qo_indptr, *args, causal=causal, **common)


@dataclass(slots=True)
class PreparedPagedAttention:
    """One FlashInfer plan shared by every physical layer in a cache group."""

    wrapper: Any
    cache: SlabPagedKvCache
    base_pages: Tensor
    physical_pages: Tensor
    layer_count: int
    tokens: int
    kv_view: tuple[int, int] | None = None

    def run(self, queries: Tensor, layer: int) -> Tensor:
        if not 0 <= layer < self.layer_count:
            raise ValueError("Qwen3.5 attention layer lies outside its cache group")
        torch.add(self.base_pages, layer, out=self.physical_pages)
        query = queries[:, 0] if self.tokens == 1 else queries.flatten(0, 1)
        keys, values = self.cache.keys, self.cache.values
        if self.kv_view is not None:
            # Another head geometry over the same KV row bytes (a drafter's).
            keys = keys.flatten(-2).unflatten(-1, self.kv_view)
            values = values.flatten(-2).unflatten(-1, self.kv_view)
        output = self.wrapper.run(query, (keys, values))
        return output.view_as(queries)


class PagedAttentionRuntime:
    """Reusable FlashInfer planning workspace for eager Qwen attention."""

    def __init__(
        self,
        device: torch.device,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        workspace_bytes: int,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.device = device
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
        self.prefill = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace, "NHD", backend="auto"
        )
        self.decode = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_tensor_cores=True,
            backend="auto",
        )

    def prepare(
        self,
        cache: SlabPagedKvCache,
        *,
        group_index: int,
        layer_count: int,
        sequence_lengths: Tensor,
        tokens: int,
        softmax_scale: float,
        heads: tuple[int, int, int] | None = None,
        causal: bool = True,
        window_left: int = -1,
    ) -> PreparedPagedAttention:
        group = cache.groups[group_index]
        page_tokens = cache.page_tokens
        query_heads, kv_heads, head_dim = heads or (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
        )
        retained_starts = group.first_blocks * page_tokens
        used_lengths = (sequence_lengths - retained_starts).to(torch.int32)
        if bool(torch.any(used_lengths <= 0)):
            raise ValueError("Qwen3.5 paged attention requires retained context")
        required_pages = torch.div(
            used_lengths + page_tokens - 1,
            page_tokens,
            rounding_mode="floor",
        )
        width = group.slab_ids.shape[1]
        if bool(torch.any(required_pages > width)):
            raise ValueError("Qwen3.5 paged attention exceeds its cache lease")

        pages_per_slab = cache.slab_units // layer_count
        page_table = group.slab_ids * pages_per_slab + group.slotnums
        columns = torch.arange(width, device=self.device)[None]
        logical_pages = page_table[columns < required_pages[:, None]].to(torch.int32)
        base_pages = logical_pages * layer_count
        physical_pages = base_pages.clone()
        kv_indptr = torch.zeros(
            len(used_lengths) + 1, dtype=torch.int32, device=self.device
        )
        kv_indptr[1:] = required_pages.cumsum(0)
        last_page_lengths = (used_lengths - 1).remainder(page_tokens) + 1
        common = {
            "pos_encoding_mode": "NONE",
            "q_data_type": torch.bfloat16,
            "kv_data_type": cache.keys.dtype,
            "o_data_type": torch.bfloat16,
            "sm_scale": softmax_scale,
            "window_left": window_left,
        }
        wrapper = self.decode if tokens == 1 else self.prefill
        _plan_wrapper(
            wrapper,
            tokens,
            kv_indptr,
            physical_pages,
            last_page_lengths,
            num_attention_heads=query_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            page_tokens=page_tokens,
            common=common,
            causal=causal,
        )
        view = None if heads is None else (kv_heads, head_dim)
        return PreparedPagedAttention(
            wrapper,
            cache,
            base_pages,
            physical_pages,
            layer_count,
            tokens,
            view,
        )


@dataclass(frozen=True, slots=True)
class GraphAttentionSpec:
    """One statically ordered attention call inside a Qwen decode graph."""

    group_index: int
    layer_count: int
    tokens: int
    sequence_offset: int
    heads: tuple[int, int, int] | None = None
    causal: bool = True
    window_left: int = -1

    def geometry(self, default: tuple[int, int, int]) -> tuple[int, int, int]:
        return self.heads or default


@dataclass(slots=True)
class _GraphAttentionCall:
    spec: GraphAttentionSpec
    prepared: PreparedPagedAttention
    kv_indptr: Tensor
    last_page_lengths: Tensor
    qo_indptr: Tensor | None


class GraphPagedAttentionRuntime:
    """Fixed-buffer FlashInfer calls used by one captured decode bucket."""

    def __init__(
        self,
        device: torch.device,
        cache: SlabPagedKvCache,
        specs: tuple[GraphAttentionSpec, ...],
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        workspace: Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.device = device
        self.cache = cache
        self.specs = specs
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.softmax_scale = head_dim**-0.5
        self.workspace = workspace
        self._cursor = 0
        calls = []
        batch_size = cache.groups[0].slab_ids.shape[0]
        default = (num_attention_heads, num_key_value_heads, head_dim)
        for spec in specs:
            query_heads, kv_heads, spec_head_dim = spec.geometry(default)
            group = cache.groups[spec.group_index]
            page_capacity = batch_size * group.slab_ids.shape[1]
            base_pages = torch.empty(page_capacity, dtype=torch.int32, device=device)
            physical_pages = torch.empty_like(base_pages)
            kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
            last = torch.empty(batch_size, dtype=torch.int32, device=device)
            common = {
                "pos_encoding_mode": "NONE",
                "q_data_type": torch.bfloat16,
                "kv_data_type": cache.keys.dtype,
                "o_data_type": torch.bfloat16,
                "sm_scale": spec_head_dim**-0.5,
                "disable_split_kv": not GRAPH_SPLIT_KV,
                "non_blocking": False,
                "window_left": spec.window_left,
            }
            initial_indptr = torch.arange(
                batch_size + 1, dtype=torch.int32, device=device
            )
            initial_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
            initial_last = torch.ones(batch_size, dtype=torch.int32, device=device)
            if spec.tokens == 1:
                wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    paged_kv_indptr_buffer=kv_indptr,
                    paged_kv_indices_buffer=physical_pages,
                    paged_kv_last_page_len_buffer=last,
                    backend="fa2",
                )
            else:
                qo_indptr = torch.empty(
                    batch_size + 1, dtype=torch.int32, device=device
                )
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=qo_indptr,
                    paged_kv_indptr_buf=kv_indptr,
                    paged_kv_indices_buf=physical_pages,
                    paged_kv_last_page_len_buf=last,
                    backend="auto",
                )
            _plan_wrapper(
                wrapper,
                spec.tokens,
                initial_indptr,
                initial_indices,
                initial_last,
                num_attention_heads=query_heads,
                num_key_value_heads=kv_heads,
                head_dim=spec_head_dim,
                page_tokens=cache.page_tokens,
                common=common,
                causal=spec.causal,
            )
            calls.append(
                _GraphAttentionCall(
                    spec,
                    PreparedPagedAttention(
                        wrapper,
                        cache,
                        base_pages,
                        physical_pages,
                        spec.layer_count,
                        spec.tokens,
                        None if spec.heads is None else (kv_heads, spec_head_dim),
                    ),
                    kv_indptr,
                    last,
                    None if spec.tokens == 1 else qo_indptr,
                )
            )
        self.calls = tuple(calls)

    def reset(self) -> None:
        self._cursor = 0

    def prepare(
        self,
        cache: SlabPagedKvCache,
        *,
        group_index: int,
        layer_count: int,
        sequence_lengths: Tensor,
        tokens: int,
        softmax_scale: float,
        heads: tuple[int, int, int] | None = None,
        causal: bool = True,
        window_left: int = -1,
    ) -> PreparedPagedAttention:
        del sequence_lengths
        if cache is not self.cache or self._cursor >= len(self.calls):
            raise RuntimeError("Qwen3.5 graph attention call order changed")
        call = self.calls[self._cursor]
        self._cursor += 1
        expected = call.spec
        if (
            expected.group_index != group_index
            or expected.layer_count != layer_count
            or expected.tokens != tokens
            or expected.heads != heads
            or expected.causal != causal
            or expected.window_left != window_left
            or softmax_scale
            != expected.geometry((0, 0, self.head_dim))[2] ** -0.5
        ):
            raise RuntimeError("Qwen3.5 graph attention signature changed")
        return call.prepared

    def graph_inputs(self) -> dict[str, Tensor]:
        return {
            f"attention.{index}.base_pages": call.prepared.base_pages
            for index, call in enumerate(self.calls)
        }

    @staticmethod
    def _host_pages(
        group: Any,
        *,
        rows: int,
        graph_rows: int,
        graph_width: int,
        sequence_lengths: Any,
        layer_count: int,
        slab_units: int,
        page_tokens: int,
    ) -> tuple[Any, Any, Any]:
        import numpy as np

        logical = np.zeros(graph_rows * graph_width, dtype=np.int32)
        indptr = np.zeros(graph_rows + 1, dtype=np.int32)
        last = np.ones(graph_rows, dtype=np.int32)
        cursor = 0
        pages_per_slab = slab_units // layer_count
        for row in range(graph_rows):
            if row < rows:
                retained = int(group.block_table_start_indices[row]) * page_tokens
                used = int(sequence_lengths[row]) - retained
                if used <= 0:
                    raise ValueError(
                        "Qwen3.5 graph attention requires retained context"
                    )
                count = (used + page_tokens - 1) // page_tokens
                slabs = group.block_slabs(row)
                slots = group.block_slots(row)
                if count > len(slabs) or count > graph_width:
                    raise ValueError("Qwen3.5 graph attention exceeds its cache lease")
                logical[cursor : cursor + count] = slabs[:count].astype(
                    np.int64
                ) * pages_per_slab + slots[:count].astype(np.int64)
                last[row] = (used - 1) % page_tokens + 1
            else:
                count = 1
            cursor += count
            indptr[row + 1] = cursor
        return logical, indptr, last

    def plan(self, cache_plan: Any, positions: Any) -> dict[str, object]:
        """Plan every wrapper for current host rows before graph replay."""

        import numpy as np

        positions = np.asarray(positions, dtype=np.int64)
        rows = cache_plan.rows
        graph_rows = self.cache.groups[0].slab_ids.shape[0]
        inputs: dict[str, object] = {}
        for index, call in enumerate(self.calls):
            spec = call.spec
            group = cache_plan.groups[spec.group_index]
            graph_width = self.cache.groups[spec.group_index].slab_ids.shape[1]
            lengths = positions + spec.sequence_offset
            logical, indptr, last = self._host_pages(
                group,
                rows=rows,
                graph_rows=graph_rows,
                graph_width=graph_width,
                sequence_lengths=lengths,
                layer_count=spec.layer_count,
                slab_units=self.cache.slab_units,
                page_tokens=self.cache.page_tokens,
            )
            physical = logical * spec.layer_count
            query_heads, kv_heads, spec_head_dim = spec.geometry(
                (self.num_attention_heads, self.num_key_value_heads, self.head_dim)
            )
            common = {
                "pos_encoding_mode": "NONE",
                "q_data_type": torch.bfloat16,
                "kv_data_type": self.cache.keys.dtype,
                "o_data_type": torch.bfloat16,
                "sm_scale": spec_head_dim**-0.5,
                "disable_split_kv": not GRAPH_SPLIT_KV,
                "non_blocking": False,
                "window_left": spec.window_left,
            }
            page_count = int(indptr[-1])
            # FlashInfer schedules on the host and copies these into its graph
            # buffers itself; device tensors would cost a sync per array.
            host_indptr = torch.from_numpy(indptr)
            host_pages = torch.from_numpy(
                np.ascontiguousarray(physical[:page_count], dtype=np.int32)
            )
            host_last = torch.from_numpy(last)
            _plan_wrapper(
                call.prepared.wrapper,
                spec.tokens,
                host_indptr,
                host_pages,
                host_last,
                num_attention_heads=query_heads,
                num_key_value_heads=kv_heads,
                head_dim=spec_head_dim,
                page_tokens=self.cache.page_tokens,
                common=common,
                causal=spec.causal,
            )
            inputs[f"attention.{index}.base_pages"] = physical
        self.reset()
        return inputs


__all__ = [
    "GraphAttentionSpec",
    "GraphPagedAttentionRuntime",
    "PagedAttentionRuntime",
    "PreparedPagedAttention",
    "append_kv",
]
