---
name: paddle-vl-throughput-bottleneck
description: "PaddleOCR-VL real throughput bottleneck — single API server (GIL), and the api_server_count patch"
metadata: 
  node_type: memory
  type: project
  originSessionId: b6356e3f-dadc-445d-9656-97d607af0e0b
---

Goal: raise PaddleOCR-VL pipeline throughput from ~0.7 to 2-3 pages/sec. Bench harness: `/workspace/bench_mp.py` (multi-process work-stealing, each worker = one PaddleOCRVL pipeline doing layout detection locally + OCR via shared vLLM server). Drive sweeps with `/workspace/sweep*.sh`. PDFs in `/workspace/papers` (500). Use FRESH offsets per run — re-running same PDFs gives fake speedups (prefix cache).

**Key debunked assumption:** the "56 req/s / 29k tok/s" from `server_stress.py` was prefix-cache-inflated (99% hit, same image repeated). Real workload = distinct region crops → **0.7% cache hit**. The earlier "3 pages/s" single-process number was also cache-inflated; real baseline is **0.59 pg/s @ 1 worker**.

**Root cause (diagnosed):** under real load the GPU is IDLE (~5% util, 130W of 700W TDP) and client workers are idle (~6% CPU), but the vLLM **API server process is pegged ~860% CPU (one process, GIL-capped)** while EngineCore (GPU) starves. Reason: paddlex's `genai/backends/vllm.py::run_vllm_server` calls vLLM's single-worker `run_server(args)`, which **ignores `api_server_count`** — so `api_server_count: 8` in `/workspace/backend_config.yaml` was a silent no-op (only 1 API server ran). Multimodal image preprocessing (decode/resize/patchify) runs in the API-server frontend and was bottlenecked on one process.

**Fix applied:** patched `…/site-packages/paddlex/inference/genai/backends/vllm.py` (backup `.orig`) to mirror vLLM's CLI dispatch — when `args.api_server_count > 1`, call `run_multi_api_server(args)` instead of `run_server`. Confirmed 8 API server processes now launch. **Why:** breaks the single-process GIL ceiling so preprocessing parallelizes across cores (96 available) and feeds the starved GPU. NOTE: patch lives in site-packages, lost if env reinstalled.

**Result after patch:** 4 workers (vl_conc 64) = **3.02 pages/sec, 0 errors** (5× the 0.59 baseline) — target met. GPU SM util rose ~5%→~28% under load. 8 workers DEGRADED to 1.63 pg/s with CV-worker OOM crashes: server holds ~115GB (0.80 util) of H200's 144GB, leaving ~29GB; each client worker's paddle layout model is ~3GB, so 8 workers (~24GB) + inference activation spikes overflow. **4 workers is the clean sweet spot.** To push higher: GPU compute is only ~28% used (headroom), so lower server `gpu-memory-utilization` (0.80→~0.70) to fit ~6-7 workers.

Other observed levers not yet exploited: vLLM uses a **slow image processor** ("use_fast is unset"); FlashInfer unavailable (sampler on torch-native fallback). Server config: gpu-mem-util 0.80, max-model-len 16384, max-num-seqs 512, api-server-count 8. See [[paddle-models-on-volume]] for paths/launch.
