#!/usr/bin/env python3
"""Inference with the distilled student (LoRA adapter) — PyTorch on MPS.

Loads the base VLM + the trained LoRA adapter, OCRs a page with the terse prompt, and parses
the pipe-delimited output into voter dicts. Used by eval_distilled.py and for spot-checks.
Run in .venv-torch.

  .venv-torch/bin/python savitr/distill/infer_torch.py --adapter models/smolvlm-savitr-lora \
      --image corpus/images/SOME_PAGE.png
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

from savitr.rolls.parse import TERSE_PROMPT, parse_terse


class DistilledOCR:
    def __init__(self, adapter, base=None, max_new_tokens=1024):
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        base = base or PeftConfig.from_pretrained(adapter).base_model_name_or_path
        model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.dev).eval()
        self.processor = AutoProcessor.from_pretrained(adapter)
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def ocr_image(self, png):
        """Return (voter dicts, generated-token count, seconds)."""
        img = Image.open(png).convert("RGB")
        msgs = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TERSE_PROMPT}]}
        ]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.dev)
        t = time.time()
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        dt = time.time() - t
        gen = out[0][inputs["input_ids"].shape[1] :]
        terse = self.processor.decode(gen, skip_special_tokens=True)
        return parse_terse(terse), int(gen.shape[0]), dt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    ocr = DistilledOCR(args.adapter, args.base, args.max_new_tokens)
    voters, ntok, dt = ocr.ocr_image(args.image)
    print(f"{len(voters)} voters, {ntok} tokens, {dt:.1f}s")
    for v in voters[:8]:
        print(
            f"  #{v['number']:<4} {v['id']:<11} {v['elector_name']:<22} "
            f"{v['relationship']} a{v['age']} {v['sex']}"
        )


if __name__ == "__main__":
    sys.exit(main())
