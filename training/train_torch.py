#!/usr/bin/env python3
"""LoRA fine-tune Qwen2-VL-2B (student) to emit terse voter rows — PyTorch, local on MPS.

MLX can't backprop through these VLMs (custom-kernel VJP gap), so training is done in
PyTorch (autodiff works for everything) on the Mac's MPS, then the merged model is converted
to MLX for fast local inference. Reuses the corpus built by build_corpus.py.

Loss is computed only on the assistant completion (the terse rows); the prompt + image tokens
are masked. Batch size 1 + grad accumulation + gradient checkpointing + frozen vision tower
to fit a 2B model on 16 GB. Run in .venv-torch.

  .venv-torch/bin/python savitr/distill/train_torch.py \
      --dataset corpus/dataset.jsonl --output models/qwen2vl-savitr-lora --epochs 2
"""

import argparse
import json
import sys
import time

import torch
from PIL import Image


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="small VLM student (2B+ models swap-thrash on 16GB)",
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default="models/qwen2vl-savitr-lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=1_000_000,
        help="cap image pixels fed to the vision tower (memory/seq control)",
    )
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {dev} | loading {args.model} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # max_pixels is a Qwen-VL processor knob; harmless to skip for others.
    proc_kw = {"max_pixels": args.max_pixels} if "qwen" in args.model.lower() else {}
    processor = AutoProcessor.from_pretrained(args.model, **proc_kw)

    # LoRA on the language-model attention/MLP; vision tower stays frozen.
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    data = load_jsonl(args.dataset)
    print(f"train examples: {len(data)}")

    def encode(ex):
        """Build input_ids + labels (loss only on the terse completion)."""
        img = Image.open(ex["image"]).convert("RGB")
        user = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ex["messages"][0]["content"]},
                ],
            }
        ]
        full = user + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": ex["messages"][1]["content"]}],
            }
        ]
        full_text = processor.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True
        )
        full_in = processor(text=[full_text], images=[img], return_tensors="pt")
        prompt_in = processor(text=[prompt_text], images=[img], return_tensors="pt")
        labels = full_in["input_ids"].clone()
        labels[:, : prompt_in["input_ids"].shape[1]] = -100  # mask prompt + image tokens
        full_in["labels"] = labels
        return {k: v.to(dev) for k, v in full_in.items()}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step = micro = 0
    t0 = time.time()
    running = 0.0
    for epoch in range(args.epochs):
        for ex in data:
            try:
                batch = encode(ex)
                loss = model(**batch).loss / args.grad_accum
                loss.backward()
            except Exception as e:  # noqa: BLE001 - skip a bad/oversized sample, keep training
                print(f"  skip sample: {e}")
                opt.zero_grad()
                continue
            running += loss.item() * args.grad_accum
            micro += 1
            if micro % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(
                        f"  epoch {epoch} step {step} loss {running/ (5*args.grad_accum):.4f} "
                        f"({(time.time()-t0)/60:.1f} min)"
                    )
                    running = 0.0
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f"\nadapter saved -> {args.output} ({step} steps, {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main())
