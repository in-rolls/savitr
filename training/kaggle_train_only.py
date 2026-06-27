#!/usr/bin/env python3
"""Kaggle GPU kernel: TRAIN-ONLY terse-Surya on a PRE-LABELED corpus.

The slow part (labeling pages with the Surya teacher) is done on the Mac with MLX (~5x faster than
PyTorch-on-T4); this kernel only LoRA-fine-tunes on the uploaded (image, terse) pairs, which fits
comfortably in Kaggle's 12h cap. Reads a corpus dataset of:
    images/*.png  +  dataset_train.jsonl  +  dataset_test.jsonl
(each jsonl line: {"image": "<path>", "messages":[{user:PROMPT},{assistant:terse}]}). Image paths are
remapped by basename to the Kaggle mount. Saves the adapter + run_report.json to /kaggle/working.

Push: kaggle kernels push --accelerator NvidiaTeslaT4  (enable_gpu + enable_internet + dataset_source)
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
from importlib.metadata import version as _ver

# Upgrade transformers for qwen3_5 (pin torch so the CUDA build isn't swapped); drop the optional
# gated-delta CUDA kernels that crash on the T4 -> pure-PyTorch path (fine: training is short-seq).
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers==5.12.1",
        "peft",
        "accelerate",
        f"torch=={_ver('torch')}",
    ],
    check=False,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "flash-linear-attention",
        "causal-conv1d",
        "flash-attn",
    ],
    check=False,
)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image

MODEL_ID = os.environ.get("SURYA_MODEL", "datalab-to/surya-ocr-2")
TERSE_PROMPT = (
    "Extract every voter from this electoral-roll page as pipe-delimited rows, "
    "one per line, columns: serial|epic|name|relation(F/H/M)|relation_name|house|age|sex"
)
TERSE_COLS = [
    "number",
    "id",
    "elector_name",
    "relationship",
    "father_or_husband_name",
    "house_no",
    "age",
    "sex",
]
MAX_PIX = int(os.environ.get("MAX_PIX", "2000000"))
_EPIC = re.compile(r"[A-Z]{1,3}\d{5,9}")
_AGE = re.compile(r"\d{1,3}")


def load_img(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIX:
        s = (MAX_PIX / (w * h)) ** 0.5
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))))
    return img


def parse_terse(text):
    """Value-anchored terse parse (robust to the model dropping the relation column)."""
    out = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        while parts and parts[0] == "":
            parts.pop(0)
        if len(parts) < 2:
            continue
        v = {c: "" for c in TERSE_COLS}
        if _EPIC.fullmatch(parts[0]):
            v["id"] = parts.pop(0)
        v["elector_name"] = parts.pop(0) if parts else ""
        if parts and parts[-1].upper() in ("M", "F", "T"):
            v["sex"] = parts.pop().upper()
        if parts and _AGE.fullmatch(parts[-1]):
            v["age"] = parts.pop()
        if parts and parts[0].upper() in ("F", "H", "M"):
            v["relationship"] = parts.pop(0).upper()
        if parts:
            v["house_no"] = parts.pop()
        if parts:
            v["father_or_husband_name"] = " ".join(parts)
        if v["elector_name"]:
            out.append(v)
    return out


def load_surya():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    return model, AutoProcessor.from_pretrained(MODEL_ID)


def gen(model, processor, img, max_new=2048):
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TERSE_PROMPT}]}
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False, eos_token_id=2, pad_token_id=0
        )
    return processor.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)


def _remap(path):
    """Corpus jsonl carries Mac abs image paths; find the same basename in the Kaggle mount."""
    hits = glob.glob(f"/kaggle/input/**/{os.path.basename(path)}", recursive=True)
    return hits[0] if hits else path


def load_corpus():
    train_jsonl = glob.glob("/kaggle/input/**/dataset_train.jsonl", recursive=True)
    test_jsonl = glob.glob("/kaggle/input/**/dataset_test.jsonl", recursive=True)

    def read(paths):
        rows = []
        for p in paths:
            for line in open(p, encoding="utf-8"):
                d = json.loads(line)
                tgt = d["messages"][1]["content"] if "messages" in d else d["target"]
                img = _remap(d["image"])
                if os.path.exists(img):
                    rows.append({"image": img, "target": tgt})
        return rows

    return read(train_jsonl), read(test_jsonl)


def train(model, processor, data, epochs=3, lr=1e-4, grad_accum=8, rank=16):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def encode(ex):
        img = load_img(ex["image"])
        user = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": TERSE_PROMPT}]}
        ]
        full = user + [{"role": "assistant", "content": [{"type": "text", "text": ex["target"]}]}]
        ft = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        pt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        fi = processor(text=[ft], images=[img], return_tensors="pt")
        pi = processor(text=[pt], images=[img], return_tensors="pt")
        labels = fi["input_ids"].clone()
        labels[:, : pi["input_ids"].shape[1]] = -100
        fi["labels"] = labels
        return {k: v.to("cuda") for k, v in fi.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    step = micro = 0
    run = 0.0
    t0 = time.time()
    for ep in range(epochs):
        for ex in data:
            torch.cuda.empty_cache()
            try:
                loss = model(**encode(ex)).loss / grad_accum
                loss.backward()
            except Exception as e:
                print(f"  skip: {e}")
                opt.zero_grad()
                continue
            run += loss.item() * grad_accum
            micro += 1
            if micro % grad_accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(
                        f"  ep {ep} step {step} loss {run/(5*grad_accum):.4f} "
                        f"({(time.time()-t0)/60:.1f} min)"
                    )
                    run = 0.0
    return model


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def _fuzzy(a, b):
    a, b = _norm(a), _norm(b)
    return a == b or (len(a) > 3 and (a in b or b in a))


def evaluate(model, processor, test):
    """OOS fidelity to the teacher: match voters by EPIC then fuzzy name; score every field."""
    model.eval()
    FIELDS = [
        "id",
        "elector_name",
        "father_or_husband_name",
        "relationship",
        "house_no",
        "age",
        "sex",
    ]
    agg = {f: [0, 0] for f in FIELDS}
    gold_n = match_n = 0
    for ex in test:
        gold = parse_terse(ex["target"])
        pred = parse_terse(gen(model, processor, load_img(ex["image"])))
        gold_n += len(gold)
        pmap = {v["id"]: v for v in pred if v.get("id")}
        used = set()
        for gv in gold:
            pv = pmap.get(gv.get("id")) if gv.get("id") else None
            if not pv or id(pv) in used:
                pv = next(
                    (
                        p
                        for p in pred
                        if id(p) not in used and _fuzzy(gv["elector_name"], p["elector_name"])
                    ),
                    None,
                )
            if not pv:
                continue
            used.add(id(pv))
            match_n += 1
            for f in FIELDS:
                if _norm(gv.get(f, "")):
                    agg[f][1] += 1
                    agg[f][0] += int(_fuzzy(gv.get(f, ""), pv.get(f, "")))
    ev = {
        "recall_pct": round(100 * match_n / max(gold_n, 1), 1),
        "gold_voters": gold_n,
        "matched": match_n,
        "fields": {},
    }
    print(f"\n=== OOS eval: {match_n}/{gold_n} = {ev['recall_pct']}% recall ===")
    for f in FIELDS:
        h, c = agg[f]
        ev["fields"][f] = round(100 * h / c, 1) if c else None
        print(f"  {f:<26} {h}/{c} = {100*h/c if c else 0:.1f}%")
    return ev


def main():
    import traceback

    rep = {"stage": "start"}
    rp = "/kaggle/working/run_report.json"

    def save():
        os.makedirs("/kaggle/working", exist_ok=True)
        json.dump(rep, open(rp, "w"), indent=2)

    out = os.environ.get("OUT_DIR", "/kaggle/working/surya-terse-lora")
    rep["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        rep["gpu"] = torch.cuda.get_device_name()
    save()
    try:
        train_set, test_set = load_corpus()
        rep["n_train"], rep["n_test"] = len(train_set), len(test_set)
        rep["stage"] = "loaded_corpus"
        save()
        print(f"corpus: {len(train_set)} train / {len(test_set)} test")
        if len(train_set) < 10:
            rep["error"] = "corpus not found / too small"
            save()
            return 1
        model, processor = load_surya()
        rep["stage"] = "model_loaded"
        save()
        _t = time.time()
        model = train(model, processor, train_set)
        rep["train_minutes"] = round((time.time() - _t) / 60, 1)
        rep["stage"] = "trained"
        save()
        if test_set:
            rep["eval"] = evaluate(model, processor, test_set)
            rep["stage"] = "evaled"
            save()
        model.save_pretrained(out)
        processor.save_pretrained(out)
        rep["stage"] = "saved"
        rep["saved"] = True
        save()
        print(f"adapter saved -> {out}")
        return 0
    except Exception as e:
        rep["error"] = str(e)
        rep["trace"] = traceback.format_exc()
        save()
        print("FATAL:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
