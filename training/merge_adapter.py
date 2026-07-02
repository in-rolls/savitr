#!/usr/bin/env python3
"""Merge the trained terse LoRA into Surya and save a standalone model for MLX conversion."""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kfin/surya-terse-lora"
OUT = sys.argv[2] if len(sys.argv) > 2 else "models/surya-terse-merged"

print(f"loading base + adapter {ADAPTER} ...")
base = AutoModelForImageTextToText.from_pretrained("datalab-to/surya-ocr-2", dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER)
print("merging ...")
model = model.merge_and_unload()
model.save_pretrained(OUT)
# Save the BASE processor/tokenizer, NOT the adapter's: the Kaggle adapter dir carries
# tokenizer/processor files from a newer transformers that the local mlx_vlm can't read
# (causes a "[gather] indices must be integral" crash at inference). The base files are canonical.
AutoProcessor.from_pretrained("datalab-to/surya-ocr-2").save_pretrained(OUT)
print(f"merged terse-Surya -> {OUT}")
