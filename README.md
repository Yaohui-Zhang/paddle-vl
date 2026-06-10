# paddle-vl

Batch PDF → Markdown/JSON parsing with **PaddleOCR-VL** on a single GPU.

A shared [vLLM](https://github.com/vllm-project/vllm) server hosts the
`PaddleOCR-VL-1.6-0.9B` vision-language model, and a multi-process,
work-stealing runner (`run.py`) feeds it PDFs in parallel. The whole thing is
driven by a single entry point: **`run.sh`**.

On an H200, the default settings parse **~144 pages/min** (`2.4 pages/sec`).

---

## What's in here

| File | Purpose |
|------|---------|
| `run.sh` | **Main entry point.** Cold-restarts the vLLM server, waits until it's ready, then runs `run.py`. All args pass straight through to `run.py`. |
| `run.py` | Multi-process runner. Spawns N workers that share the vLLM server and pull PDFs off a work queue, saving per-PDF Markdown + JSON. |
| `backend_config.yaml` | vLLM server tuning (GPU mem fraction, max model len, API server count, etc). |
| `fetch_papers.py` | Helper to download ~200 random arXiv PDFs into `/workspace/papers` for testing. |
| `Dockerfile` | The image behind the RunPod template (CUDA 12.8 + PyTorch + PaddlePaddle-GPU + PaddleOCR + vLLM, plus a paddlex patch for multi–API-server throughput). |

---

## 1. Deploy a pod

Click the template link and deploy a pod:

👉 **https://console.runpod.io/deploy?template=qyrn6j1gom&ref=6s9jsb79**

- Pick a GPU. A single **NVIDIA H200** (or any ≥24 GB GPU) is plenty — the model
  is only 0.9B params. More VRAM mainly lets you raise concurrency.
- Make sure your SSH public key is set in your RunPod account settings (so the
  pod authorizes your key), then launch.

The template ships the prebuilt image with all dependencies (PaddleOCR, vLLM,
PyTorch, FlashAttention) already installed — nothing to `pip install`.

## 2. SSH into the pod

Once the pod is **Running**, copy its SSH command from the RunPod dashboard
(Connect → SSH), e.g.:

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/id_ed25519
```

## 3. Clone the repo

```bash
cd /workspace
git clone https://github.com/Yaohui-Zhang/paddle-vl.git
cd paddle-vl
```

## 4. Parse PDFs with `run.sh`

Point `run.sh` at one or more PDF files and/or directories of PDFs:

```bash
# parse every *.pdf in a directory (defaults: 4 workers, vl-conc 64)
./run.sh /workspace/papers

# parse specific files
./run.sh paper1.pdf paper2.pdf

# choose an output directory
./run.sh /workspace/papers -o /workspace/output
```

**First run downloads the model.** On the very first invocation the model dir
is missing, so the server auto-downloads `PaddleOCR-VL-1.6` (and the layout
model `PP-DocLayoutV3`) into `/workspace/.cache/paddle/` — this adds ~1 minute.
Every later run finds the cache and skips the download.

> Need test PDFs? Run `python3 fetch_papers.py` to pull ~200 random arXiv
> papers into `/workspace/papers`.

### Output

For each input PDF, results land in `<output-dir>/<pdf-name>/` (default
`output/`), with one Markdown + JSON file **per page**, plus an `imgs/` folder
of extracted figures:

```
output/
└── 2605.00966v1/
    ├── 2605.00966v1_0.md
    ├── 2605.00966v1_0.json
    ├── 2605.00966v1_1.md
    ├── ...
    └── imgs/
```

When the run finishes you'll see a throughput summary:

```
==== 4 workers | 271 pages / 113.0s = 2.398 pages/sec (143.9 pg/min) | errs=0 ====
```

---

## Parameters

`run.sh [pdfs/dirs ...] [options]` — options are forwarded verbatim to `run.py`:

| Flag | Default | Meaning |
|------|---------|---------|
| `-w`, `--workers N` | `4`* | Number of worker **processes**. Each loads the layout model and pulls PDFs off a shared queue. |
| `-c`, `--vl-conc N` | `64` | Max concurrent VL recognition requests **per worker** sent to the vLLM server. |
| `-o`, `--output-dir DIR` | `output` | Where per-PDF Markdown + JSON is written. |
| `--server URL` | `http://localhost:8118/v1` | vLLM server URL (the one `run.sh` starts locally). |

\* `run.py`'s own default is 8, but **`--workers 4 --vl-conc 64` is the
recommended setting** and what we benchmark with — it saturates the GPU without
oversubscribing. Use it explicitly:

```bash
./run.sh /workspace/papers --workers 4 --vl-conc 64
```

**Tuning notes**
- `workers` controls CPU-side parallelism (PDF rasterization, layout detection,
  pre/post-processing). Too many oversubscribes CPU; too few starves the GPU.
  **4 is the sweet spot** on a single H200.
- `vl-conc` controls how many recognition requests each worker keeps in flight
  to the vLLM server. **64** keeps the GPU busy at 4 workers. Raise it only if
  the GPU is under-utilized; it's bounded by the server's `max-num-seqs` (512 in
  `backend_config.yaml`).
- The total in-flight load on the server is roughly `workers × vl-conc`.

---

## How it works

```
run.sh
 ├─ kill any old vLLM server, wait for GPU memory to free
 ├─ start `paddleocr genai_server` (vLLM backend) on :8118
 │     using backend_config.yaml  ── waits until /v1/models responds
 └─ exec run.py
        ├─ spawn N worker processes (shared vLLM server, work-stealing queue)
        ├─ each worker: PDF → layout (PP-DocLayoutV3) → VL recognition (vLLM)
        │              → restructure → save .md / .json / imgs per page
        └─ print pages/sec summary
```

`run.sh` **always cold-restarts** the server (kills `genai_server` /
`EngineCore`, waits for GPU memory to drop below 5 GB) so every run starts from
a clean GPU state.

---

## Troubleshooting

- **`SERVER TIMEOUT`** — the vLLM server didn't come up in time. Check the
  server log: `cat /workspace/vllm_server.log`. First-run model download can be
  slow; `run.sh` already extends the timeout to 30 min when the model cache is
  missing.
- **Server won't start / GPU busy** — `run.sh` waits for GPU memory to fall
  below 5 GB before starting. If something else is holding the GPU, free it
  first (`nvidia-smi`).
- **Logs**:
  - `run.sh` / `run.py` output → wherever you redirect stdout (e.g. the terminal).
  - vLLM server → `/workspace/vllm_server.log`.
