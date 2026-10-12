#!/usr/bin/env python3
"""Environment check: CUDA, bf16 matmul, flash-attn kernel, library versions,
the Hugging Face cache in use, and (unless --skip-model) a Qwen3-VL-8B load +
forward.

    python scripts/check_env.py
    python scripts/check_env.py --model artifacts/models/Qwen3-VL-8B-Instruct
    python scripts/check_env.py --skip-model
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
p.add_argument("--model", default="artifacts/models/Qwen3-VL-8B-Instruct",
               help="Base model dir to load (relative paths are resolved against the repo root).")
p.add_argument("--skip-model", action="store_true", help="Skip the model load + forward.")
args = p.parse_args()

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA not available")
for i in range(torch.cuda.device_count()):
    print(f"device {i}: {torch.cuda.get_device_name(i)} | capability {torch.cuda.get_device_capability(i)} "
          f"| {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB")
print("compiled archs:", torch.cuda.get_arch_list())

a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = (a @ a).float().abs().mean().item()
torch.cuda.synchronize()
print(f"bf16 matmul OK (mean {b:.3f})")

try:
    import flash_attn
    from flash_attn import flash_attn_func
    q = torch.randn(2, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, q, q, causal=True)
    torch.cuda.synchronize()
    print("flash_attn", flash_attn.__version__, "kernel forward OK", tuple(out.shape))
except Exception as e:  # noqa: BLE001
    print("flash_attn NOT usable:", e, "\n  -> set attn_implementation: sdpa in the configs (or --attn sdpa)")

import transformers, peft, trl, deepspeed, accelerate  # noqa: E402
print("transformers", transformers.__version__, "| peft", peft.__version__,
      "| trl", trl.__version__, "| deepspeed", deepspeed.__version__,
      "| accelerate", accelerate.__version__)
from huggingface_hub import constants as hf_constants  # noqa: E402
print("HF hub cache:", hf_constants.HF_HUB_CACHE,
      "(set HF_HOME to relocate; HF_HUB_OFFLINE=1 once everything is cached)")
from transformers import Qwen3VLForConditionalGeneration  # noqa: E402

if not args.skip_model:
    path = args.model if os.path.isabs(args.model) else str(REPO_ROOT / args.model)
    if not os.path.isdir(path):
        print(f"base model not found at {path}; run scripts/download_assets.sh (--skip-model to skip)")
    else:
        attn = "flash_attention_2" if "flash_attn" in sys.modules else "sdpa"
        print(f"loading {path} with attn_implementation={attn} ...")
        m = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation=attn, device_map={"": 0})
        ids = torch.arange(16, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = m(input_ids=ids).logits
        print("Qwen3-VL-8B forward OK", tuple(logits.shape),
              f"| peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print("CHECK PASS")
