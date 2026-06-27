#!/usr/bin/env python3
"""Kaggle GPU kernel: fine-tune Surya itself to emit TERSE voter rows.

Self-contained (Kaggle can't import our package). On a free T4 it: loads datalab-to/surya-ocr-2,
labels the uploaded page PNGs with Surya (OCR->HTML->parse->terse, quality-filtered), LoRA
fine-tunes Surya to map page->terse rows, evals on a held-out split, and saves the adapter to
/kaggle/working. Output is a ~40 MB LoRA adapter to download + run locally on MLX.

Inputs : /kaggle/input/<dataset>/  (PNG pages from prep_pages.py)
Output : /kaggle/working/surya-terse-lora/  (LoRA adapter) + eval printed to log.

Pushed headlessly via `kaggle kernels push` (enable_gpu + enable_internet).
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
from importlib.metadata import version as _ver

# Kaggle ships an older transformers without qwen3_5. Upgrade transformers (internet is enabled),
# but PIN torch to the pre-installed CUDA build — letting pip pull a different torch breaks the T4
# ("CUDA error: no kernel image is available for execution on the device").
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
# Gated-delta FAST path: install flash-linear-attention (Triton -> JIT-compiles for the T4's sm_75,
# so it works where the precompiled causal-conv1d/flash-attn -- sm_80+ -- crash). The v8 "fla broke
# import torch" was actually an empty-kernel bug (now fixed), so fla gets a fair test. Pin torch so
# pip can't swap the working CUDA build. If fla's API mismatches transformers, run_report's traceback
# shows it (transformers also auto-falls-back to pure-PyTorch if fla is simply absent).
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "causal-conv1d", "flash-attn"], check=False
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "flash-linear-attention",
        f"torch=={_ver('torch')}",
    ],
    check=False,
)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from PIL import Image

MODEL_ID = os.environ.get("SURYA_MODEL", "datalab-to/surya-ocr-2")
OCR_PROMPT = "OCR this image to HTML."
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

# ---- inlined parsing helpers (from savitr/mlx_ocr.py) ------------------------
TAG = re.compile(r"<[^>]+>")
EPIC = re.compile(r"\b([A-Z]{2,3}\d{6,8})\b")
NAME = re.compile(r"Name\s*:\s*(.*?)(?:Father|Husband|Mother|House|Age|Gender|$)", re.I | re.S)
REL = re.compile(
    r"(Father|Husband|Mother)'?s?\s*Name\s*:\s*(.*?)(?:House|Age|Gender|$)", re.I | re.S
)
HOUSE = re.compile(r"House\s*Number\s*:\s*(.*?)(?:Age|Gender|$)", re.I | re.S)
AGE = re.compile(r"Age\s*:\s*(\d{1,3})", re.I)
GENDER = re.compile(r"Gender\s*:\s*(Male|Female|Third|Other)", re.I)
REL_CODE = {"father": "F", "husband": "H", "mother": "M"}
SEX_CODE = {"male": "M", "female": "F", "third": "T", "other": "T"}


def _clean(s):
    return re.sub(r"\s+", " ", TAG.sub(" ", s)).strip(" :")


def _is_header(html, m):
    pre = html[max(0, m.start() - 16) : m.start()]
    post = html[m.end() : m.end() + 24]
    return bool(
        re.search(r"(Father|Husband|Mother)'?s?\s*$", pre, re.I)
        or re.search(r"\band\s*$", pre, re.I)
        or re.match(r"\s*and\s+Reservation", post, re.I)
    )


def parse_voters(html):
    name_iters = [m for m in re.finditer(r"Name\s*:", html, re.I) if not _is_header(html, m)]
    voters, starts = [], []
    for i, m in enumerate(name_iters):
        blob = html[
            m.start() : (name_iters[i + 1].start() if i + 1 < len(name_iters) else len(html))
        ]
        nm = NAME.search(blob)
        name = _clean(nm.group(1)) if nm else ""
        am, gm = AGE.search(blob), GENDER.search(blob)
        if not name or name[0].isdigit() or not (am or gm):
            continue
        rel, hm = REL.search(blob), HOUSE.search(blob)
        pre = html[max(0, m.start() - 28) : m.start()]
        lead = re.search(r"(#)?\s*(\d{1,4})\s*(?:<br\s*/?>)\s*(?:<[^>]*>\s*)*$", pre)
        starts.append(m.start())
        voters.append(
            {
                "elector_name": name,
                "father_or_husband_name": _clean(rel.group(2)) if rel else "",
                "relationship": REL_CODE.get(rel.group(1).lower(), "") if rel else "",
                "house_no": _clean(hm.group(1)) if hm else "",
                "age": am.group(1) if am else "",
                "sex": SEX_CODE.get(gm.group(1).lower(), "") if gm else "",
                "number": lead.group(2) if lead else "",
            }
        )
    seen, cut = set(), len(voters)
    for j, v in enumerate(voters):
        key = (re.sub(r"\s+", " ", v["elector_name"].lower()).strip(), v["age"])
        if key in seen:
            cut = j
            break
        seen.add(key)
    clean = html[: starts[cut]] if cut < len(voters) else html
    voters = voters[:cut]
    epics = EPIC.findall(clean)
    bold = re.findall(r"<b>\s*(#)?\s*(\d{1,4})\s*</b>", clean)
    for k, v in enumerate(voters):
        v["id"] = epics[k] if k < len(epics) else ""
        if not v["number"] and k < len(bold):
            v["number"] = bold[k][1]
    return voters


def dedupe_voters(voters):
    def score(v):
        return sum(
            1
            for k in ("id", "number", "father_or_husband_name", "house_no", "age", "sex")
            if v.get(k)
        )

    best = {}
    for v in voters:
        key = (
            ("epic", v["id"])
            if v.get("id")
            else ("id", re.sub(r"\s+", " ", v["elector_name"].lower()).strip(), v.get("age", ""))
        )
        if key not in best or score(v) > score(best[key]):
            best[key] = v
    return sorted(
        best.values(), key=lambda v: int(v["number"]) if v.get("number", "").isdigit() else 1e9
    )


def to_terse(voters):
    san = lambda x: re.sub(r"[|\r\n]+", " ", str(x)).strip()
    return "\n".join("|".join(san(v.get(c, "")) for c in TERSE_COLS) for v in voters)


def parse_terse(text):
    out = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        p = [x.strip() for x in line.split("|")] + [""] * 8
        v = dict(zip(TERSE_COLS, p))
        if v["elector_name"]:
            out.append(v)
    return out


def is_clean(voters):
    n = len(voters)
    if not (20 <= n <= 35):
        return False
    epics = [v["id"] for v in voters if v.get("id")]
    if len(epics) < 0.9 * n or len(set(epics)) != len(epics):
        return False
    return sum(1 for v in voters if v.get("age") and v.get("sex")) >= 0.9 * n


# ---- model + generation ------------------------------------------------------
MAX_PIX = int(
    os.environ.get("MAX_PIX", "2000000")
)  # cap page pixels so the vision tower fits the T4


def load_img(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > MAX_PIX:  # 192-DPI pages are ~3.9 MP -> ~20k patches -> OOM
        s = (MAX_PIX / (w * h)) ** 0.5
        img = img.resize((max(8, int(w * s)), max(8, int(h * s))))
    return img


def load_surya():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def gen(model, processor, img, prompt, max_new=8192):  # full OCR pages run ~7700 tokens
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp = processor(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False, eos_token_id=2, pad_token_id=0
        )  # generation_config.json (config.json's 248044 is stale)
    return processor.decode(out[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)


# ---- 1) label the uploaded pages with Surya ----------------------------------
def build_dataset(model, processor, page_dir, test_every=10):
    allp = sorted(glob.glob(os.path.join(page_dir, "**", "*.png"), recursive=True))
    cap = int(os.environ.get("MAX_PAGES", "40"))  # bound T4 time; stride-sample for AC diversity
    stride = max(1, len(allp) // cap)
    pngs = allp[::stride][:cap]
    cap_min = int(
        os.environ.get("MAX_LABEL_MIN", "480")
    )  # stop labeling in time to train+save <12h
    print(
        f"labeling {len(pngs)}/{len(allp)} pages with Surya (cap={cap}, stride={stride}, "
        f"time_cap={cap_min}min) ..."
    )
    train, test = [], []
    t0 = time.time()
    for i, png in enumerate(pngs):
        if (time.time() - t0) / 60 > cap_min:
            print(
                f"  hit {cap_min}min labeling cap at page {i}; stopping to leave time to train+save"
            )
            break
        torch.cuda.empty_cache()
        try:
            img = load_img(png)
            voters = dedupe_voters(parse_voters(gen(model, processor, img, OCR_PROMPT)))
        except Exception as e:
            print(f"  skip {os.path.basename(png)}: {e}")
            continue
        if not is_clean(voters):
            continue
        rec = {"image": png, "target": to_terse(voters)}
        (
            test if len(train) + len(test) and (len(train) + len(test)) % test_every == 0 else train
        ).append(rec)
        if (i + 1) % 10 == 0:
            print(
                f"  {i+1}/{len(pngs)} seen, kept {len(train)+len(test)} "
                f"({(time.time()-t0)/60:.1f} min, {(time.time()-t0)/(i+1):.0f}s/page)"
            )
    print(f"kept {len(train)} train / {len(test)} test")
    return train, test


# ---- 2) LoRA fine-tune Surya to terse ----------------------------------------
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
            try:
                loss = model(**encode(ex)).loss / grad_accum
                loss.backward()
            except Exception as e:
                print(f"  skip sample: {e}")
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
                        f"  ep {ep} step {step} loss {run/(5*grad_accum):.4f} ({(time.time()-t0)/60:.1f} min)"
                    )
                    run = 0.0
    return model


# ---- 3) eval on the held-out gold test split ---------------------------------
def evaluate(model, processor, test):
    model.eval()
    FIELDS = ["id", "elector_name", "father_or_husband_name", "house_no", "age", "sex"]
    hit = {f: [0, 0] for f in FIELDS}
    gold_n = match_n = 0
    toks = []
    for ex in test:
        gold = {v["number"]: v for v in parse_terse(ex["target"]) if v.get("number")}
        out = gen(model, processor, load_img(ex["image"]), TERSE_PROMPT)
        toks.append(len(processor.tokenizer(out)["input_ids"]))
        pred = {v["number"]: v for v in parse_terse(out) if v.get("number")}
        gold_n += len(gold)
        for k in set(gold) & set(pred):
            match_n += 1
            for f in FIELDS:
                a = gold[k][f].lower().strip()
                if a:
                    hit[f][1] += 1
                    b = pred[k][f].lower().strip()
                    hit[f][0] += int(
                        a == b
                        or (
                            f in ("elector_name", "father_or_husband_name")
                            and b
                            and (a in b or b in a)
                        )
                    )
    avg_tok = sum(toks) / max(len(toks), 1)
    print(f"\n=== EVAL (terse-Surya vs Surya gold) ===")
    print(f"avg {avg_tok:.0f} tokens/page (vs ~6-8k HTML) | recall {match_n}/{gold_n}")
    ev = {"avg_tokens_per_page": round(avg_tok, 1), "recall": [match_n, gold_n], "fields": {}}
    for f in FIELDS:
        h, c = hit[f]
        print(f"  {f:<24} {h}/{c}  {100*h/c if c else 0:.1f}%")
        ev["fields"][f] = {"hit": h, "total": c, "pct": round(100 * h / c, 1) if c else None}
    return ev


def main():
    import traceback

    rep = {"stage": "start"}  # written to /kaggle/working so `output` downloads it
    rp = "/kaggle/working/run_report.json"

    def save():
        os.makedirs("/kaggle/working", exist_ok=True)
        json.dump(rep, open(rp, "w"), indent=2)

    inp = os.environ.get("PAGE_DIR") or "/kaggle/input"
    out = os.environ.get("OUT_DIR", "/kaggle/working/surya-terse-lora")
    print(f"device cuda={torch.cuda.is_available()} | pages={inp}")
    rep["cuda"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        rep["gpu"] = torch.cuda.get_device_name()
        rep["cap"] = list(torch.cuda.get_device_capability())
        rep["torch"] = torch.__version__
        try:
            (torch.randn(8, 8, device="cuda") @ torch.randn(8, 8, device="cuda")).sum().item()
            rep["matmul"] = "OK"
            print("cuda matmul OK")
        except Exception as e:
            rep["matmul"] = f"FAIL: {e}"
            print(f"cuda matmul FAILED: {e}")
    save()
    try:
        model, processor = load_surya()
        rep["stage"] = "loaded"
        save()
        _t = time.time()
        train_set, test_set = build_dataset(model, processor, inp)
        rep["label_minutes"] = round((time.time() - _t) / 60, 1)  # fla fast-path vs pure-torch tell
        rep["n_train"], rep["n_test"] = len(train_set), len(test_set)
        rep["stage"] = "labeled"
        save()
        if len(train_set) < 5:
            rep["error"] = "too few clean pages"
            save()
            print("too few clean pages; aborting")
            return 1
        model = train(model, processor, train_set)
        rep["stage"] = "trained"
        save()
        rep["eval"] = evaluate(model, processor, test_set)
        rep["stage"] = "evaled"
        save()
        model.save_pretrained(out)
        processor.save_pretrained(out)
        json.dump(test_set, open(os.path.join(out, "test_split.json"), "w"))
        rep["stage"] = "saved"
        rep["saved"] = True
        save()
        print(f"\nadapter saved -> {out}")
        return 0
    except Exception as e:
        rep["error"] = str(e)
        rep["trace"] = traceback.format_exc()
        save()
        print("FATAL:\n" + traceback.format_exc())
        raise


if __name__ == "__main__":
    sys.exit(main())
