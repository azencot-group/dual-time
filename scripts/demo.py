#!/usr/bin/env python3
"""Ask the dual-tower model about one synthetic time series: a TimeSeriesExam-style
multiple-choice question (the example of the model card).

    python scripts/demo.py                      # released adapters under artifacts/checkpoints/DualTime-8B
    python scripts/demo.py --stage-a DIR --stage-b DIR
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from scipy import signal
from peft import PeftModel
from src import paths
from src.model.model_patch import load_qwen3vl_dual
from src.datasets_utils.unified_collator import UnifiedDualCollator
from src.datasets_utils.prompts import SYSTEM, user_text

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default="artifacts/models/Qwen3-VL-8B-Instruct", help="base model dir or Hub id")
p.add_argument("--stage-a", default="artifacts/checkpoints/DualTime-8B/stageA/best", help="Stage A prior adapter")
p.add_argument("--stage-b", default="artifacts/checkpoints/DualTime-8B/stageB/checkpoint-5714", help="Stage B LoRA adapter")
args = p.parse_args()

model, processor = load_qwen3vl_dual(paths.resolve(args.model))        # Qwen3-VL + DINOv3 delay tower, bf16
model = model.to("cuda")
model = PeftModel.from_pretrained(model, paths.resolve(args.stage_a)).merge_and_unload()   # 1. Stage A prior
model = PeftModel.from_pretrained(model, paths.resolve(args.stage_b)).merge_and_unload()   # 2. Stage B LoRA
model.eval()
collator = UnifiedDualCollator(processor=processor, max_length=4096, chart_style="v2")


def ask(series, question, system=None, max_new_tokens=256):
    # "<ts><ts/>" marks where the series goes; the collator renders it as the line plot
    # for the Qwen ViT and the delay-embedding image for the DINOv3 tower.
    example = {"system": system, "user_text": "<ts><ts/>\n" + question, "target_text": "",
               "ts_list": [np.asarray(series, dtype=np.float32)], "ts_field_names": ["series"],
               "task": "demo", "meta": {}}
    inputs = collator([example], for_generation=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    return processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


t = np.arange(512)
series = signal.sawtooth(2 * np.pi * t / 40) + 2 * signal.square(2 * np.pi * t / 128)
question = "Which additive combination of patterns best describes the time series?"
options = ["SineWave + SquareWave", "SawtoothWave + SquareWave", "SineWave + SawtoothWave"]
print("Answer:", ask(series, user_text(question, options), system=SYSTEM, max_new_tokens=8))
