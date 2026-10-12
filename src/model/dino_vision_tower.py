"""DINOv3 ViT-L/16 vision tower with Qwen-style mergers.

Honours the contract `Qwen3VLModel.get_video_features` expects from a tower:

    embeds, deepstack_embeds = tower(pixel_values, grid_thw=grid_thw)
    split_sizes = (grid_thw.prod(-1) // tower.spatial_merge_size**2).tolist()

  - `.spatial_merge_size` = 2 (matching Qwen3-VL);
  - `embeds` shaped `(sum_per_image(H*W) // merge**2, out_hidden_size)`,
    rows ordered image by image;
  - `deepstack_embeds` = list of tensors (one per tap layer) with the same shape.

Input: raw pixels `(N, 3, H, W)` with H, W multiples of the patch size (16).
For 256x256 delay images that is 16x16 = 256 patches -> 64 merged tokens.

Three intermediate layer outputs (`deepstack_indexes`) go through separate
post-shuffle mergers, exactly like `deepstack_merger_list` in Qwen3VLVisionModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class DinoVisionTowerConfig:
    hf_model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    patch_size: int = 16
    spatial_merge_size: int = 2
    out_hidden_size: int = 4096
    deepstack_indexes: Tuple[int, int, int] = (6, 12, 18)
    image_size: int = 256
    freeze_backbone: bool = True


class _PatchMerger(nn.Module):
    """Mirror of Qwen3VLVisionPatchMerger: norm -> fc1 -> gelu -> fc2 (-> out_hidden).

    Adapted from `transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchMerger`
    (Hugging Face transformers, Apache License 2.0); same parameter names so a
    Qwen3-VL merger checkpoint loads into it.

    Input: shuffled tokens shaped (M, in_hidden * merge_size**2); each row is
    the channel-wise concat of the merge_size**2 base patch tokens merged into
    one output token.
    """

    def __init__(self, in_hidden: int, merge_size: int, out_hidden: int,
                 use_postshuffle_norm: bool):
        super().__init__()
        self.in_hidden = in_hidden
        self.merge_size = merge_size
        self.shuffled = in_hidden * (merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        if use_postshuffle_norm:
            self.norm = nn.LayerNorm(self.shuffled, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(in_hidden, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.shuffled, self.shuffled)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.shuffled, out_hidden)

    def forward(self, shuffled_tokens: torch.Tensor) -> torch.Tensor:
        x = shuffled_tokens
        if self.use_postshuffle_norm:
            x = self.norm(x)
        else:
            M = x.shape[0]
            x = x.view(M, -1, self.in_hidden)  # (M, m^2, in_hidden)
            x = self.norm(x).reshape(M, self.shuffled)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


def _block_shuffle(tokens_b_hw_c: torch.Tensor, grid_h: int, grid_w: int,
                   merge_size: int) -> torch.Tensor:
    """(B, H*W, C) -> (B * (H/m) * (W/m), C * m*m), m x m sub-blocks packed contiguously."""
    B, N, C = tokens_b_hw_c.shape
    assert N == grid_h * grid_w, f"got {N} tokens for {grid_h}x{grid_w} grid"
    m = merge_size
    x = tokens_b_hw_c.view(B, grid_h, grid_w, C)
    x = x.view(B, grid_h // m, m, grid_w // m, m, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H/m, W/m, m, m, C)
    return x.view(B * (grid_h // m) * (grid_w // m), C * m * m)


class DinoVisionTower(nn.Module):
    """DINOv3 ViT backbone + Qwen-style mergers, mimicking Qwen3VLVisionModel."""

    _no_split_modules: List[str] = ["DINOv3ViTLayer"]

    def __init__(self, cfg: DinoVisionTowerConfig):
        super().__init__()
        from transformers import DINOv3ViTModel

        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.out_hidden_size = cfg.out_hidden_size
        self.image_size = cfg.image_size

        self.backbone = DINOv3ViTModel.from_pretrained(cfg.hf_model_id)
        # An out-of-range tap would capture fewer deepstack features than there
        # are mergers, and the LM-side zip would then silently drop the last
        # injection for BOTH towers. Fail loudly instead.
        _n_layers = self.backbone.config.num_hidden_layers
        _bad = [i for i in cfg.deepstack_indexes if not (0 <= int(i) < _n_layers)]
        if _bad:
            raise ValueError(
                f"deepstack_indexes {_bad} out of range for a {_n_layers}-layer "
                f"DINO backbone (valid: 0..{_n_layers - 1})")
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden = self.backbone.config.hidden_size
        self.base_hidden = hidden
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        # Tokens per image after the spatial merge: (256/16/2)^2 = 64.
        self.tokens_per_image = (cfg.image_size // cfg.patch_size // cfg.spatial_merge_size) ** 2

        self.merger = _PatchMerger(
            in_hidden=hidden, merge_size=cfg.spatial_merge_size,
            out_hidden=cfg.out_hidden_size, use_postshuffle_norm=False)
        self.deepstack_indexes = list(cfg.deepstack_indexes)
        self.deepstack_merger_list = nn.ModuleList([
            _PatchMerger(in_hidden=hidden, merge_size=cfg.spatial_merge_size,
                         out_hidden=cfg.out_hidden_size, use_postshuffle_norm=True)
            for _ in self.deepstack_indexes
        ])

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode even when the Trainer calls
        model.train() on the whole tree (guards against train/eval skew if a
        backbone with dropout / drop-path is ever used)."""
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.backbone.eval()
        return self

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def _strip_special_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Drop CLS + register tokens, leaving only the patch tokens."""
        skip = 1 + self.num_register_tokens
        return hidden_states[:, skip:, :]

    def _forward_backbone(self, pixel_values: torch.Tensor):
        """Run DINOv3 layer by layer, collecting outputs at the tap layers and the final norm."""
        bb = self.backbone
        pv = pixel_values.to(bb.embeddings.patch_embeddings.weight.dtype)
        hidden_states = bb.embeddings(pv, bool_masked_pos=None)
        position_embeddings = bb.rope_embeddings(pv)

        captured: List[torch.Tensor] = []
        deepstack_set = set(self.deepstack_indexes)
        # The layer list lives at `bb.layer` (transformers 4.x) or `bb.model.layer` (5.x).
        bb_layers = getattr(bb, "layer", None)
        if bb_layers is None:
            bb_layers = bb.model.layer
        for i, layer_module in enumerate(bb_layers):
            hidden_states = layer_module(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
            )
            if i in deepstack_set:
                captured.append(hidden_states)

        hidden_states = bb.norm(hidden_states)
        return hidden_states, captured

    def forward(self, pixel_values: torch.Tensor,
                grid_thw: Optional[torch.LongTensor] = None, **kwargs):
        """pixel_values: (B, 3, H, W). grid_thw: (B, 3) rows [1, H/patch, W/patch]."""
        if pixel_values.dim() != 4:
            raise ValueError(
                f"DinoVisionTower expects (B, 3, H, W) pixel_values, got {tuple(pixel_values.shape)}")

        B, _, H, W = pixel_values.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        m = self.spatial_merge_size
        if ph % m != 0 or pw % m != 0:
            raise ValueError(f"Patch grid {ph}x{pw} not divisible by spatial_merge_size={m}")

        if grid_thw is None:
            grid_thw = torch.tensor([[1, ph, pw]] * B, dtype=torch.long, device=pixel_values.device)
        if grid_thw.shape[0] != B:
            raise ValueError(f"grid_thw rows ({grid_thw.shape[0]}) != batch images ({B})")

        final, captured = self._forward_backbone(pixel_values)
        final_patches = self._strip_special_tokens(final)
        patch_embeds = self.merger(_block_shuffle(final_patches, ph, pw, m))   # (B * P, out_hidden)

        deepstack_features = []
        for tap, merger in zip(captured, self.deepstack_merger_list):
            patches = self._strip_special_tokens(tap)
            deepstack_features.append(merger(_block_shuffle(patches, ph, pw, m)))
        return patch_embeds, deepstack_features
