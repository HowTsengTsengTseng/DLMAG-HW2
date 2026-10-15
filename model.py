"""PyTorch Transformer-XL adapted to REMI token sequences.

Architecture follows YatingMusic/remi's Transformer-XL: sinusoidal relative
positions, content/position attention biases, relative shift, post-norm blocks,
and cached layer inputs. The forward and memory API matches train.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_head: int = 8
    d_inner: int = 2048
    n_layer: int = 12
    segment_len: int = 512
    memory_len: int = 512
    dropout: float = 0.1
    architecture: str = "remi-transformer-xl-v2"

    def validate(self) -> None:
        if min(self.vocab_size, self.d_model, self.n_head, self.d_inner,
               self.n_layer, self.segment_len, self.memory_len) < 1:
            raise ValueError("Model dimensions and sequence lengths must be positive")
        if self.d_model % self.n_head or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.architecture != "remi-transformer-xl-v2":
            raise ValueError("Unsupported model architecture")


class SinusoidalRelativePosition(nn.Module):
    """Distances from oldest memory token down to the current token."""

    def __init__(self, d_model: int):
        super().__init__()
        inverse = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def forward(self, key_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        positions = torch.arange(key_len - 1, -1, -1, device=device, dtype=torch.float32)
        phase = torch.outer(positions, self.inverse_frequency.to(device))
        return torch.cat((phase.sin(), phase.cos()), dim=-1).to(dtype)


def relative_shift(values: Tensor) -> Tensor:
    """Align relative position scores with [memory, current segment] keys."""
    batch, heads, query_len, key_len = values.shape
    padded = F.pad(values, (1, 0))
    shifted = padded.reshape(batch, heads, key_len + 1, query_len)
    return shifted[:, :, 1:, :].reshape(batch, heads, query_len, key_len)


class RelativeAttention(nn.Module):
    """Transformer-XL content and relative-position attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.relative_key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attention_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: Tensor, memory: Tensor, memory_valid: Tensor,
        input_valid: Tensor, position: Tensor, content_bias: Tensor,
        position_bias: Tensor,
    ) -> Tensor:
        batch, query_len, width = x.shape
        memory_len = memory.shape[1]
        source = torch.cat((memory, x), dim=1)
        key_len = source.shape[1]

        projected = self.qkv(source).reshape(
            batch, key_len, 3, self.n_head, self.head_dim
        )
        query = projected[:, -query_len:, 0].transpose(1, 2)
        key = projected[:, :, 1].transpose(1, 2)
        value = projected[:, :, 2].transpose(1, 2)
        relative_key = self.relative_key(position).reshape(
            key_len, self.n_head, self.head_dim
        ).transpose(0, 1)

        content = torch.einsum(
            "bhtd,bhkd->bhtk", query + content_bias[None, :, None].to(query.dtype), key
        )
        relative = torch.einsum(
            "bhtd,hkd->bhtk", query + position_bias[None, :, None].to(query.dtype),
            relative_key,
        )
        scores = (content + relative_shift(relative)) / math.sqrt(self.head_dim)

        query_positions = memory_len + torch.arange(query_len, device=x.device)
        key_positions = torch.arange(key_len, device=x.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        key_valid = torch.cat((memory_valid, input_valid), dim=1)
        allowed = causal[None, None] & key_valid[:, None, None, :]
        scores = scores.float().masked_fill(~allowed, torch.finfo(torch.float32).min)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1)).to(value.dtype)
        attended = torch.matmul(weights, value).transpose(1, 2)
        attended = attended.reshape(batch, query_len, width)
        return self.out(attended)


class TransformerXLBlock(nn.Module):
    """Post-norm attention and feed-forward block, matching the reference."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = RelativeAttention(cfg)
        self.attention_norm = nn.LayerNorm(cfg.d_model)
        self.inner = nn.Linear(cfg.d_model, cfg.d_inner)
        self.outer = nn.Linear(cfg.d_inner, cfg.d_model)
        self.feedforward_norm = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: Tensor, memory: Tensor, memory_valid: Tensor,
        input_valid: Tensor, position: Tensor, content_bias: Tensor,
        position_bias: Tensor,
    ) -> Tensor:
        attended = self.attention(
            x, memory, memory_valid, input_valid, position, content_bias, position_bias
        )
        x = self.attention_norm(x + self.dropout(attended))
        feedforward = self.dropout(F.relu(self.inner(x)))
        feedforward = self.dropout(F.relu(self.outer(feedforward)))
        return self.feedforward_norm(x + feedforward)


class TransformerXL(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = SinusoidalRelativePosition(cfg.d_model)
        head_dim = cfg.d_model // cfg.n_head
        self.content_bias = nn.Parameter(torch.empty(cfg.n_head, head_dim))
        self.position_bias = nn.Parameter(torch.empty(cfg.n_head, head_dim))
        self.blocks = nn.ModuleList(TransformerXLBlock(cfg) for _ in range(cfg.n_layer))
        self.dropout = nn.Dropout(cfg.dropout)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.lm_head.weight = self.embedding.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.content_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.position_bias, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)
        for block in self.blocks:
            nn.init.normal_(block.attention.qkv.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.attention.relative_key.weight, mean=0.0, std=0.01)
            nn.init.normal_(block.attention.out.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.inner.weight, mean=0.0, std=0.02)
            nn.init.normal_(block.outer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(block.inner.bias)
            nn.init.zeros_(block.outer.bias)

    def empty_memory(self, batch: int, device: torch.device) -> tuple[list[Tensor], Tensor]:
        memories = [
            torch.zeros(batch, self.cfg.memory_len, self.cfg.d_model, device=device)
            for _ in self.blocks
        ]
        valid = torch.zeros(batch, self.cfg.memory_len, dtype=torch.bool, device=device)
        return memories, valid

    def forward(
        self, ids: Tensor, input_valid: Tensor, memories: list[Tensor], memory_valid: Tensor
    ) -> tuple[Tensor, list[Tensor], Tensor]:
        if ids.ndim != 2 or ids.shape != input_valid.shape:
            raise ValueError("ids and input_valid must have matching [batch, length] shapes")
        if ids.shape[1] > self.cfg.segment_len:
            raise ValueError("Input exceeds segment_len")
        if len(memories) != self.cfg.n_layer:
            raise ValueError("Expected one memory tensor per transformer layer")

        key_len = self.cfg.memory_len + ids.shape[1]
        position = self.position(key_len, ids.device, self.embedding.weight.dtype)
        position = self.dropout(position)
        x = self.dropout(self.embedding(ids) * math.sqrt(self.cfg.d_model))
        layer_inputs = []
        for block, memory in zip(self.blocks, memories, strict=True):
            layer_inputs.append(x)
            x = block(
                x, memory, memory_valid, input_valid, position,
                self.content_bias, self.position_bias,
            )
        logits = self.lm_head(self.dropout(x))
        new_memory, new_valid = self._advance_memory(
            memories, memory_valid, layer_inputs, input_valid
        )
        return logits, new_memory, new_valid

    @torch.no_grad()
    def _advance_memory(
        self, memories: list[Tensor], old_valid: Tensor, layer_inputs: list[Tensor],
        input_valid: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        # Store each layer's input, detached, with real tokens right-aligned.
        batch = input_valid.shape[0]
        keep = self.cfg.memory_len
        next_valid = torch.zeros_like(old_valid)
        next_memory = [torch.zeros_like(memory) for memory in memories]
        for lane in range(batch):
            prior_count = int(old_valid[lane].sum().item())
            current_count = int(input_valid[lane].sum().item())
            take = min(keep, prior_count + current_count)
            next_valid[lane, keep - take:] = True
            for layer in range(len(memories)):
                prior = (memories[layer][lane, keep - prior_count:]
                         if prior_count else memories[layer][lane, :0])
                current = layer_inputs[layer][lane, :current_count].detach()
                next_memory[layer][lane, keep - take:] = torch.cat((prior, current), dim=0)[-take:]
        return next_memory, next_valid
