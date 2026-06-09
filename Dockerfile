FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Base system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Unify the python / pip commands
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch CUDA 12.8
RUN python -m pip install \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# PaddlePaddle GPU. CUDA is backward compatible within 12.x, so the cu126
# build runs fine on a 12.8 driver. (The cu128 index ships a broken/CPU wheel.)
RUN python -m pip install \
    paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/


# Prebuilt FlashAttention
RUN python -m pip install \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl

# Install PaddleOCR. Pin paddlex too: it's pulled transitively, and the
# api-server-count patch below is tied to paddlex's source layout.
RUN python -m pip install "paddleocr[doc-parser]==3.6.0" "paddlex==3.6.1"

# Install vLLM inference dependencies.
# There is no GPU driver during `docker build` (libcuda.so.1 is injected at
# runtime by nvidia-container-toolkit), but this command imports paddle which
# needs libcuda.so.1. We temporarily point it at the CUDA toolkit's stub lib to
# satisfy the load, then remove the stub so it never shadows the real runtime
# driver. The stub lets `import paddle` succeed, but the command still probes the
# GPU at the end and the stub driver always returns cudaErrorStubLibrary. The pip
# deps are already installed before that probe, so we tolerate the probe failure,
# then verify vllm is actually installed (without triggering CUDA) and fail the
# build if it is not. At runtime the real driver is present, so the probe passes.
RUN mkdir -p /tmp/culink && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/culink/libcuda.so.1 && \
    LD_LIBRARY_PATH=/tmp/culink:$LD_LIBRARY_PATH paddleocr install_genai_server_deps vllm \
      || echo "[build] GPU probe failed under stub driver (expected without real GPU); deps already installed"; \
    rm -rf /tmp/culink && \
    python -m pip install "vllm==0.10.2" && \
    python -c "import importlib.metadata as m; print('vllm installed:', m.version('vllm'))"

# Patch paddlex so `api-server-count > 1` actually launches multiple vLLM frontend
# API servers. Upstream's run_vllm_server() always calls the single-worker
# run_server(args) and silently ignores api_server_count, so multimodal image
# preprocessing is GIL-capped on one process and the GPU starves (~1 pg/s instead
# of ~2.3 pg/s at 4 workers). We mirror vLLM's own CLI dispatch. Located via
# importlib.metadata (NO import of paddlex/paddle — there's no GPU at build time).
# Idempotent; FAILS THE BUILD if the anchor is gone, so a paddlex/vllm version
# bump can't silently revert the fix.
RUN python - <<'PY'
import importlib.metadata as md, pathlib, sys
target = next(f for f in md.files("paddlex")
              if str(f).replace("\\", "/").endswith("inference/genai/backends/vllm.py"))
path = pathlib.Path(md.distribution("paddlex").locate_file(target))
src = path.read_text()
if "run_multi_api_server" in src:
    print("[patch] already applied:", path); sys.exit(0)
anchor = "    uvloop.run(run_server(args))"
if anchor not in src:
    sys.exit(f"[patch] ERROR: anchor not found in {path}; paddlex layout changed, update the patch")
new = (
    "    if getattr(args, 'api_server_count', 1) > 1:\n"
    "        from vllm.entrypoints.cli.serve import run_multi_api_server\n"
    "        run_multi_api_server(args)\n"
    "    else:\n"
    "        uvloop.run(run_server(args))"
)
path.write_text(src.replace(anchor, new))
print("[patch] applied api_server_count dispatch to", path)
PY

WORKDIR /workspace

# SSH server (RunPod provides SSH access via $PUBLIC_KEY + sshd)
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Startup script: configure SSH + export env vars + keep the container alive
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Keep the container alive and run sshd so the pod doesn't restart / SSH works
CMD ["/start.sh"]