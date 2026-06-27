"""An MLX inference backend for Surya — Apple Silicon, ~3.6× faster than llama.cpp.

Surya ships `vllm` (NVIDIA) and `llamacpp` (CPU/Apple Silicon) backends. This adds a third,
`mlx`, that serves the MLX-converted model through mlx-vlm's OpenAI-compatible server and
plugs into the exact same `SuryaInferenceManager` / `chat_completions_batch` path — so the
full Surya pipeline (layout, full-page OCR, table rec) runs unchanged, just faster.

Design mirrors `LlamaCppBackend`: spawn an OpenAI server, attach an OpenAI client, delegate
generation to Surya's shared `chat_completions_batch`. The mlx-vlm server runs as its own
subprocess (its own venv), so it never conflicts with Surya's dependencies.

Usage (drop-in, no fork needed):
    import mlx_backend; mlx_backend.register()
    os.environ["SURYA_MLX_MODEL_PATH"] = "models/surya-mlx-4bit"
    os.environ["SURYA_MLX_PYTHON"] = ".venv-mlx/bin/python"   # python that has mlx-vlm
    manager = SuryaInferenceManager(method="mlx")

Convert the model once:
    .venv-mlx/bin/python -m mlx_vlm convert --hf-path datalab-to/surya-ocr-2 \
        --mlx-path models/surya-mlx-4bit -q --q-bits 4

Upstreaming: this file is the body of `surya/inference/backends/mlx.py`; `register()` is the
two-line addition to `surya.inference._build_backend` (add an `"mlx"` branch).
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI
from surya.inference.backends.base import Backend, ServerHandle
from surya.inference.backends.openai_client import chat_completions_batch
from surya.inference.schema import BatchInputItem, BatchOutputItem
from surya.logging import get_logger

logger = get_logger()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class MlxBackend(Backend):
    """Surya inference backend that serves the model via mlx-vlm's OpenAI server."""

    name = "mlx"

    def __init__(self):
        self.handle: ServerHandle | None = None
        self._client: OpenAI | None = None
        self._proc: subprocess.Popen | None = None

    # ---- config (env, mirroring SURYA_INFERENCE_* conventions) ----
    @property
    def _model_path(self) -> str:
        p = os.environ.get("SURYA_MLX_MODEL_PATH")
        if not p:
            raise RuntimeError("Set SURYA_MLX_MODEL_PATH to the converted MLX model dir.")
        return os.path.abspath(p)

    def start(self) -> ServerHandle:
        """Spawn (or reuse) the mlx-vlm server and return its handle."""
        if self.handle is not None:
            return self.handle

        host = os.environ.get("SURYA_INFERENCE_HOST", "127.0.0.1")

        # Attach to an externally-managed server if pinned.
        external = os.environ.get("SURYA_MLX_URL") or os.environ.get("SURYA_INFERENCE_URL")
        if external:
            base = external.rstrip("/")
            base = base if base.endswith("/v1") else base + "/v1"
            model_name = self._probe_model_id(base) or self._model_path
        else:
            model_path = self._model_path
            port = int(os.environ.get("SURYA_MLX_PORT", "0")) or _free_port()
            mlx_python = os.environ.get("SURYA_MLX_PYTHON", sys.executable)
            cmd = [
                mlx_python,
                "-m",
                "mlx_vlm",
                "server",
                "--model",
                model_path,
                "--host",
                host,
                "--port",
                str(port),
            ]
            # Speculative decoding via Surya's native MTP head — byte-identical output, ~1.14×.
            draft = os.environ.get("SURYA_MLX_DRAFT_MODEL")
            if draft:
                cmd += [
                    "--draft-model",
                    os.path.abspath(draft),
                    "--draft-kind",
                    os.environ.get("SURYA_MLX_DRAFT_KIND", "mtp"),
                ]
            for extra in (os.environ.get("SURYA_MLX_EXTRA_ARGS", "")).split():
                cmd.append(extra)
            log_path = Path("~/.cache/datalab/surya/mlx_server.log").expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Spawning mlx-vlm server: {' '.join(cmd)}")
            self._proc = subprocess.Popen(
                cmd, stdout=open(log_path, "ab"), stderr=subprocess.STDOUT, start_new_session=True
            )
            atexit.register(self.stop)
            base = f"http://{host}:{port}/v1"
            self._wait_ready(base)
            # mlx-vlm registers the model under the exact --model string it was given.
            model_name = model_path

        self.handle = ServerHandle(
            base_url=base, model_name=model_name, spawned_by_us=self._proc is not None
        )
        self._client = OpenAI(api_key="EMPTY", base_url=base)
        return self.handle

    def _wait_ready(self, base: str, timeout: float = 300.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"mlx-vlm server exited early (code {self._proc.returncode}); "
                    f"see ~/.cache/datalab/surya/mlx_server.log"
                )
            try:
                if httpx.get(f"{base}/models", timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("mlx-vlm server did not become ready in time.")

    @staticmethod
    def _probe_model_id(base: str) -> str | None:
        try:
            data = httpx.get(f"{base}/models", timeout=5).json()
            return data["data"][0]["id"]
        except Exception:
            return None

    def stop(self) -> None:
        """Terminate the mlx-vlm server and clear cached client/handle."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
        self.handle = None
        self._client = None

    def generate(self, batch: list[BatchInputItem]) -> list[BatchOutputItem]:
        """Run a batch of chat completions against the running server."""
        if self.handle is None or self._client is None:
            self.start()
        assert self.handle is not None  # start() populates handle + client
        return chat_completions_batch(
            batch,
            client=self._client,
            model_name=self.handle.model_name,
            timeout=float(os.environ.get("SURYA_INFERENCE_TIMEOUT_SECONDS", "600")),
            # MLX/Metal is memory-bandwidth bound: concurrent dense pages thrash, so keep the
            # default low (see FINDINGS.md). Override with SURYA_INFERENCE_PARALLEL.
            max_workers=int(os.environ.get("SURYA_INFERENCE_PARALLEL", "2")),
            # mlx-vlm logprobs support varies; off by default (confidence -> 1.0).
            request_logprobs_default=os.environ.get("SURYA_MLX_LOGPROBS", "0") == "1",
        )


def register() -> None:
    """Teach `SuryaInferenceManager(method='mlx')` to build this backend.

    Upstream, this is just an `if method == 'mlx': return MlxBackend()` branch in
    `surya.inference._build_backend`. Here we patch it so no fork is required.
    """
    import surya.inference as inf

    _orig = inf._build_backend

    def _build(method: str):
        if method.lower() == "mlx":
            return MlxBackend()
        return _orig(method)

    inf._build_backend = _build
