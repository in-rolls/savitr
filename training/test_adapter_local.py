#!/usr/bin/env python3
"""Quick local sanity test: run the trained LoRA on a held-out test page, print output vs gold.
Shows whether terse-Surya learned anything (independent of the serial-keyed eval metric)."""

import glob
import json
import os

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ADAPTER = "/tmp/kfin/surya-terse-lora"
PAGES = "/Users/soodoku/Documents/GitHub/savitr/pages"
TERSE_PROMPT = (
    "Extract every voter from this electoral-roll page as pipe-delimited rows, "
    "one per line, columns: serial|epic|name|relation(F/H/M)|relation_name|house|age|sex"
)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
test = json.load(open(ADAPTER + "/test_split.json"))
print(f"device={dev} | {len(test)} test examples")

model = AutoModelForImageTextToText.from_pretrained("datalab-to/surya-ocr-2", dtype=torch.float32)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.to(dev).eval()
proc = AutoProcessor.from_pretrained(ADAPTER)

for ex in test[:2]:
    base = os.path.basename(ex["image"])
    loc = glob.glob(f"{PAGES}/{base}")
    if not loc:
        print(f"\n[{base}] not found locally, skip")
        continue
    img = Image.open(loc[0]).convert("RGB")
    w, h = img.size
    if w * h > 2_000_000:
        s = (2_000_000 / (w * h)) ** 0.5
        img = img.resize((int(w * s), int(h * s)))
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TERSE_PROMPT}]}
    ]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=900, do_sample=False, eos_token_id=2, pad_token_id=0
        )
    res = proc.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"\n===== {base} =====")
    print("--- MODEL OUTPUT (first 1200 chars) ---")
    print(res[:1200])
    print("--- GOLD (first 600 chars) ---")
    print(ex["target"][:600])
