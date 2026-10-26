"""Fungible full-attention, DeltaNet-state, and optional MTP cache."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from resin import CacheGroupFamily, CacheGroupLayout, CacheLayout
from resin.backend.cuda.slab import (
    DevicePageGroup as DeviceGroup,
)
from resin.backend.cuda.slab import (
    GraphPageTableArena,
    SlabDeviceCache,
    stage_page_groups,
)
from resin.backend.plans import PagedCachePlan
from resin.backend.resources import CachePoolBinding
from resin.models.qwen3_5.config import Qwen35Config

PAGE_TOKENS = 128
PREFILL_CHUNK_SIZE = 4096
TARGET_GROUP = 0
STATE_GROUP = 1
MTP_GROUP = 2

BF16_BYTES = 2
FP32_BYTES = 4


@dataclass(frozen=True, slots=True)
class CacheGeometry:
    """Common slab geometry for unequal Qwen cache resources."""

    full_layers: int
    linear_layers: int
    kv_heads: int
    head_dim: int
    linear_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    conv_dim: int
    conv_state_length: int
    hidden_size: int
    include_mtp: bool
    kv_element_bytes: int
    slab_units: int
    target_blocks_per_slab: int
    state_blocks_per_slab: int
    mtp_blocks_per_slab: int

    @classmethod
    def from_config(
        cls,
        config: Qwen35Config,
        *,
        include_mtp: bool,
        kv_element_bytes: int = BF16_BYTES,
        draft_layers: int = 1,
    ) -> CacheGeometry:
        """`draft_layers` sizes the MTP group's blocks: 1 for the MTP head,
        or a block drafter's layer count (whose KV rows share these bytes)."""
        if include_mtp:
            config.require_mtp()
        if kv_element_bytes not in {1, 2}:
            raise ValueError("Qwen3.5 KV elements must occupy one or two bytes")
        conv_state_length = config.linear_conv_kernel_dim - 1
        unit_bytes = (
            2
            * PAGE_TOKENS
            * config.num_key_value_heads
            * config.head_dim
            * kv_element_bytes
        )
        target_block_bytes = config.full_layer_count * unit_bytes
        state_bytes = (
            config.linear_layer_count
            * config.linear_num_value_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim
            * FP32_BYTES
            + config.linear_layer_count
            * config.linear_conv_dim
            * conv_state_length
            * BF16_BYTES
            + config.hidden_size * BF16_BYTES
        )
        # Slabs must divide evenly into both groups' layer-major blocks.
        lcm_layers = math.lcm(config.full_layer_count, draft_layers)
        block_bytes = lcm_layers * unit_bytes
        slab_bytes = math.ceil(state_bytes / block_bytes) * block_bytes
        slab_units = slab_bytes // unit_bytes
        return cls(
            full_layers=config.full_layer_count,
            linear_layers=config.linear_layer_count,
            kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            linear_value_heads=config.linear_num_value_heads,
            linear_key_head_dim=config.linear_key_head_dim,
            linear_value_head_dim=config.linear_value_head_dim,
            conv_dim=config.linear_conv_dim,
            conv_state_length=conv_state_length,
            hidden_size=config.hidden_size,
            include_mtp=include_mtp,
            kv_element_bytes=kv_element_bytes,
            slab_units=slab_units,
            target_blocks_per_slab=slab_units // config.full_layer_count,
            state_blocks_per_slab=slab_bytes // state_bytes,
            mtp_blocks_per_slab=slab_units // draft_layers,
        )

    @property
    def unit_elements(self) -> int:
        return 2 * PAGE_TOKENS * self.kv_heads * self.head_dim

    @property
    def unit_bytes(self) -> int:
        return self.unit_elements * self.kv_element_bytes

    @property
    def slab_bytes(self) -> int:
        return self.slab_units * self.unit_bytes

    @property
    def recurrent_elements(self) -> int:
        return (
            self.linear_layers
            * self.linear_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
        )

    @property
    def recurrent_bytes(self) -> int:
        return self.recurrent_elements * FP32_BYTES

    @property
    def convolution_elements(self) -> int:
        return self.linear_layers * self.conv_dim * self.conv_state_length

    @property
    def state_bytes(self) -> int:
        return (
            self.recurrent_bytes
            + self.convolution_elements * BF16_BYTES
            + self.hidden_size * BF16_BYTES
        )

    def layout(self, slab_capacity: int) -> CacheLayout:
        groups = [
            CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.target_blocks_per_slab),
            CacheGroupLayout(
                PREFILL_CHUNK_SIZE,
                blocks_per_slab=self.state_blocks_per_slab,
                family=CacheGroupFamily.STATE,
            ),
        ]
        if self.include_mtp:
            groups.append(
                CacheGroupLayout(PAGE_TOKENS, blocks_per_slab=self.mtp_blocks_per_slab)
            )
        return CacheLayout(parent_capacity=slab_capacity, groups=groups)


@dataclass(frozen=True, slots=True)
class DeviceCache(SlabDeviceCache):
    """Tensorized cache tables and typed views over the shared byte arena."""

    geometry: CacheGeometry
    groups: tuple[DeviceGroup, ...]
    keys: Tensor
    values: Tensor
    recurrent_states: Tensor
    convolution_states: Tensor
    target_hidden_states: Tensor

    def state_at(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions, block_tokens=PREFILL_CHUNK_SIZE
        )
        indices = slabs.long(), slots.long()
        return (
            self.recurrent_states[indices],
            self.convolution_states[indices],
            self.target_hidden_states[indices],
        )

    def store_state(
        self,
        positions: Tensor,
        recurrent: Tensor,
        convolution: Tensor,
        target_hidden: Tensor,
    ) -> None:
        slabs, slots, _ = self.locations(
            STATE_GROUP, positions, block_tokens=PREFILL_CHUNK_SIZE
        )
        indices = slabs.long(), slots.long()
        self.recurrent_states[indices] = recurrent
        self.convolution_states[indices] = convolution
        self.target_hidden_states[indices] = target_hidden


class CacheArena(GraphPageTableArena):
    """Layer-addressable BF16 KV and mixed-dtype recurrent state views."""

    graph_label = "Qwen3.5"

    def __init__(
        self,
        device: Any,
        layout: CacheLayout,
        geometry: CacheGeometry,
        *,
        kv_dtype: Any = None,
    ) -> None:
        self.device = device
        self.layout = layout
        self.geometry = geometry
        if kv_dtype is None:
            kv_dtype = torch.bfloat16
        if kv_dtype.itemsize != geometry.kv_element_bytes:
            raise ValueError("Qwen3.5 KV dtype does not match its cache geometry")
        self.kv_dtype = kv_dtype
        self.binding = CachePoolBinding(layout, resource="Qwen3.5 cache")
        physical = layout.parent_capacity + 1
        self.storage = torch.zeros(
            (physical, geometry.slab_bytes), dtype=torch.uint8, device=device
        )

        kv = self.storage.view(kv_dtype)
        bf16 = self.storage.view(torch.bfloat16)
        page_elements = PAGE_TOKENS * geometry.kv_heads * geometry.head_dim
        page_shape = (
            physical * geometry.slab_units,
            PAGE_TOKENS,
            geometry.kv_heads,
            geometry.head_dim,
        )
        page_stride = (
            geometry.unit_elements,
            geometry.kv_heads * geometry.head_dim,
            geometry.head_dim,
            1,
        )
        self.keys = torch.as_strided(kv, page_shape, page_stride)
        self.values = torch.as_strided(
            kv, page_shape, page_stride, storage_offset=page_elements
        )

        fp32 = self.storage.view(torch.float32)
        recurrent_shape = (
            physical,
            geometry.state_blocks_per_slab,
            geometry.linear_layers,
            geometry.linear_value_heads,
            geometry.linear_key_head_dim,
            geometry.linear_value_head_dim,
        )
        recurrent_layer = (
            geometry.linear_value_heads
            * geometry.linear_key_head_dim
            * geometry.linear_value_head_dim
        )
        self.recurrent_states = torch.as_strided(
            fp32,
            recurrent_shape,
            (
                geometry.slab_bytes // FP32_BYTES,
                geometry.state_bytes // FP32_BYTES,
                recurrent_layer,
                geometry.linear_key_head_dim * geometry.linear_value_head_dim,
                geometry.linear_value_head_dim,
                1,
            ),
        )

        convolution_layer = geometry.conv_dim * geometry.conv_state_length
        self.convolution_states = torch.as_strided(
            bf16,
            (
                physical,
                geometry.state_blocks_per_slab,
                geometry.linear_layers,
                geometry.conv_dim,
                geometry.conv_state_length,
            ),
            (
                geometry.slab_bytes // BF16_BYTES,
                geometry.state_bytes // BF16_BYTES,
                convolution_layer,
                geometry.conv_state_length,
                1,
            ),
            storage_offset=geometry.recurrent_bytes // BF16_BYTES,
        )
        self.target_hidden_states = torch.as_strided(
            bf16,
            (
                physical,
                geometry.state_blocks_per_slab,
                geometry.hidden_size,
            ),
            (
                geometry.slab_bytes // BF16_BYTES,
                geometry.state_bytes // BF16_BYTES,
                1,
            ),
            storage_offset=(
                geometry.recurrent_bytes // BF16_BYTES + geometry.convolution_elements
            ),
        )

    def reset_null(self) -> None:
        self.storage[0].zero_()

    def tensorize(self, plan: PagedCachePlan, staging: Any) -> DeviceCache:
        return DeviceCache(
            self.geometry,
            stage_page_groups(plan, staging),
            self.keys,
            self.values,
            self.recurrent_states,
            self.convolution_states,
            self.target_hidden_states,
        )

    def _make_graph_cache(self, groups) -> DeviceCache:
        return DeviceCache(
            self.geometry,
            groups,
            self.keys,
            self.values,
            self.recurrent_states,
            self.convolution_states,
            self.target_hidden_states,
        )


__all__ = [
    "CacheArena",
    "CacheGeometry",
    "DeviceCache",
    "DeviceGroup",
    "MTP_GROUP",
    "PAGE_TOKENS",
    "PREFILL_CHUNK_SIZE",
    "STATE_GROUP",
    "TARGET_GROUP",
]
