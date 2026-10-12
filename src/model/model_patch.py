"""Load Qwen3-VL with a second, DINOv3-backed vision tower (the "dual tower").

    from src.model.model_patch import load_qwen3vl_dual
    model, processor = load_qwen3vl_dual(qwen_path="artifacts/models/Qwen3-VL-8B-Instruct")

The native Qwen ViT (`model.model.visual`) is kept intact and serves the
*image* stream (matplotlib charts). The DINOv3 tower is attached as
`model.model.visual_dino` and serves the *video* stream (delay-embedding
images) through a patched `get_video_features`. Everything else in the
Qwen3-VL forward (placeholder masks, M-RoPE, deepstack merge) is reused.
"""

from __future__ import annotations

import types
from typing import Optional

import torch

from .dino_vision_tower import DinoVisionTower, DinoVisionTowerConfig

DEFAULT_DINO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _dual_get_video_features(self, pixel_values_videos, video_grid_thw=None):
    """`Qwen3VLModel.get_video_features` routed through `self.visual_dino`.
    `pixel_values_videos` is `(N, 3, H, W)` raw pixels (see UnifiedDualCollator)."""
    pv = pixel_values_videos.type(self.visual_dino.dtype)
    video_embeds, deepstack_video_embeds = self.visual_dino(pv, grid_thw=video_grid_thw)
    split_sizes = (video_grid_thw.prod(-1) // self.visual_dino.spatial_merge_size ** 2).tolist()
    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, deepstack_video_embeds


def load_qwen3vl_dual(
    qwen_path: str,
    dino_id: str = DEFAULT_DINO_ID,
    image_size: int = 256,
    spatial_merge_size: int = 2,
    deepstack_indexes=(6, 12, 18),
    freeze_dino_backbone: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    device_map: Optional[str] = None,
):
    """Load Qwen3-VL with both vision towers active. Returns (model, processor)."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    kwargs = dict(dtype=dtype, attn_implementation=attn_implementation)
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = Qwen3VLForConditionalGeneration.from_pretrained(qwen_path, **kwargs)
    processor = AutoProcessor.from_pretrained(qwen_path)

    # The forward zips the image and video deepstack lists, so both towers
    # must inject at the same number of LM layers.
    native_deepstack = list(model.config.vision_config.deepstack_visual_indexes)
    if len(deepstack_indexes) != len(native_deepstack):
        raise ValueError(
            f"DINO deepstack count ({len(deepstack_indexes)}) must match the native "
            f"Qwen ViT deepstack count ({len(native_deepstack)}).")

    cfg = DinoVisionTowerConfig(
        hf_model_id=dino_id,
        patch_size=16,
        spatial_merge_size=spatial_merge_size,
        out_hidden_size=int(model.config.vision_config.out_hidden_size),
        deepstack_indexes=tuple(deepstack_indexes),
        image_size=image_size,
        freeze_backbone=freeze_dino_backbone,
    )
    dino = DinoVisionTower(cfg)
    dino.to(dtype=dtype)
    try:
        dino.to(device=next(model.model.visual.parameters()).device)
    except StopIteration:
        pass

    # Register as a submodule (PEFT / state dict / device placement see it)
    # and re-point the video stream at it.
    model.model.visual_dino = dino
    model.model.get_video_features = types.MethodType(_dual_get_video_features, model.model)
    return model, processor
