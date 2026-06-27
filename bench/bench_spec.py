#!/usr/bin/env python3
"""Benchmark speculative decoding for Surya: speed + IDENTICAL-output check.

Spawns the mlx-vlm OpenAI server with a given (model [, draft-model]) config, OCRs N real
pages greedily (temp=0), and records each page's text + token count + wall time. Run it once
without a draft (baseline) and once with the MTP draft, then diff the texts (must be identical
— that's the accuracy guarantee) and compare speed. Run in .venv-mlx.

  .venv-mlx/bin/python bench/bench_spec.py PDF --pages 3-5 --model models/surya-mlx-4bit \
      [--draft-model models/surya-mtp-draft --draft-kind mtp] --out /tmp/base.json
"""

import argparse
import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

PROMPT = "OCR this image to HTML."


def free_port():
    s = socket.socket(); s.bind(("", 0)); p = s.getsockname()[1]; s.close(); return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--pages", default="3-5")
    ap.add_argument("--dpi", type=int, default=192)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--draft-kind", default="mtp")
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pdf2image import convert_from_path
    from pdf2image.pdf2image import pdfinfo_from_path

    npages = int(pdfinfo_from_path(args.pdf)["Pages"])
    idx = []
    for part in args.pages.split(","):
        if "-" in part:
            a, b = part.split("-"); idx += range(int(a) - 1, int(b))
        else:
            idx.append(int(part) - 1)
    idx = [i for i in idx if 0 <= i < npages]
    imgs = []
    for i in idx:
        im = convert_from_path(args.pdf, dpi=args.dpi, first_page=i + 1, last_page=i + 1)[0].convert("RGB")
        buf = io.BytesIO(); im.save(buf, format="PNG")
        imgs.append(base64.b64encode(buf.getvalue()).decode())

    port = free_port()
    cmd = [sys.executable, "-m", "mlx_vlm", "server", "--model", os.path.abspath(args.model),
           "--host", "127.0.0.1", "--port", str(port)]
    if args.draft_model:
        cmd += ["--draft-model", os.path.abspath(args.draft_model), "--draft-kind", args.draft_kind]
    label = "SPEC(" + args.draft_kind + ")" if args.draft_model else "BASELINE"
    print(f"[{label}] spawning: {' '.join(cmd[3:])}")
    log = open(f"/tmp/spec_server_{port}.log", "wb")
    srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    base_url = f"http://127.0.0.1:{port}/v1"
    model_id = os.path.abspath(args.model)
    try:
        for _ in range(180):
            try:
                if urllib.request.urlopen(f"{base_url}/models", timeout=2).status == 200:
                    break
            except Exception:
                pass
            if srv.poll() is not None:
                raise RuntimeError(f"server exited early; see /tmp/spec_server_{port}.log")
            time.sleep(2)

        pages = []
        for n, img in enumerate(imgs):
            body = {"model": model_id, "temperature": 0.0, "max_tokens": args.max_tokens,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]}
            req = urllib.request.Request(f"{base_url}/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            dt = time.time() - t
            txt = r["choices"][0]["message"]["content"] or ""
            tok = r.get("usage", {}).get("completion_tokens", 0)
            pages.append({"page": idx[n] + 1, "text": txt, "tokens": tok, "seconds": round(dt, 1)})
            print(f"  page {idx[n]+1}: {tok} tok, {dt:.1f}s, {tok/dt if dt else 0:.1f} tok/s")
    finally:
        try:
            os.killpg(os.getpgid(srv.pid), 15)
        except Exception:
            srv.terminate()

    tot_tok = sum(p["tokens"] for p in pages); tot_s = sum(p["seconds"] for p in pages)
    out = {"label": label, "model": args.model, "draft": args.draft_model,
           "tok_per_s": round(tot_tok / tot_s, 1) if tot_s else 0,
           "s_per_page": round(tot_s / len(pages), 1), "pages": pages}
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"\n[{label}] {out['tok_per_s']} tok/s | {out['s_per_page']} s/page -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
