#!/usr/bin/env python3
"""LoRA fine-tune the MLX Surya model on the local terse-voter corpus.

Thin launcher that reuses mlx-vlm's own training internals (lora.setup_model_for_training,
trainer.VisionDataset, trainer.sft_trainer.train) but loads our LOCAL jsonl (mlx-vlm's
lora.py CLI only takes HF dataset ids). Run in .venv-mlx.

  .venv-mlx/bin/python savitr/distill/train.py \
      --model-path models/surya-mlx-bf16 --dataset corpus/dataset.jsonl \
      --output models/savitr-lora --iters 800

NOTE: max-seq-length must hold image (~3.5k tokens) + prompt + terse target (~4.2k total);
the mlx-vlm default of 2048 would truncate and break training.
"""

import argparse
import sys

import mlx.optimizers as optim
from datasets import Image, load_dataset
from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train
from mlx_vlm.utils import load


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", default="models/surya-mlx-bf16")
    ap.add_argument("--dataset", required=True, help="local dataset.jsonl")
    ap.add_argument("--output", default="models/savitr-lora", help="adapter output dir")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-seq-length", type=int, default=6144)
    ap.add_argument("--image-resize-shape", type=int, nargs=2, default=None)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--train-vision", action="store_true")
    ap.add_argument("--full-finetune", action="store_true")
    ap.add_argument("--train-on-completions", action="store_true")
    ap.add_argument("--assistant-id", type=int, default=77091)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--steps-per-report", type=int, default=10)
    ap.add_argument("--steps-per-eval", type=int, default=100)
    ap.add_argument("--steps-per-save", type=int, default=100)
    args = ap.parse_args()

    out = (
        args.output
        if args.output.endswith(".safetensors")
        else args.output + "/adapters.safetensors"
    )

    print(f"loading model {args.model_path} ...")
    model, processor = load(args.model_path, processor_config={"trust_remote_code": True})
    config = model.config.__dict__

    print(f"loading dataset {args.dataset} ...")
    ds = load_dataset("json", data_files=args.dataset, split="train").cast_column("image", Image())
    n_val = max(1, int(len(ds) * args.val_frac)) if len(ds) > 10 else 0
    val_ds = train_ds = None
    if n_val:
        split = ds.train_test_split(test_size=n_val, seed=7)
        ds_train, ds_val = split["train"], split["test"]
    else:
        ds_train, ds_val = ds, None
    ds_train = transform_dataset_to_messages(ds_train, config.get("model_type"))
    train_ds = VisionDataset(
        ds_train, config, processor, image_resize_shape=args.image_resize_shape
    )
    if ds_val is not None:
        ds_val = transform_dataset_to_messages(ds_val, config.get("model_type"))
        val_ds = VisionDataset(ds_val, config, processor)
    print(f"train={len(ds_train)}  val={len(ds_val) if ds_val is not None else 0}")

    model = setup_model_for_training(model, args)

    targs = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.steps_per_save,
        max_seq_length=args.max_seq_length,
        adapter_file=out,
        grad_checkpoint=args.grad_checkpoint,
        learning_rate=args.learning_rate,
        full_finetune=args.full_finetune,
    )
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=targs,
        train_on_completions=args.train_on_completions,
        assistant_id=args.assistant_id,
    )
    print(f"\nadapters saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
