#!/usr/bin/env python3
"""Multi-process PaddleOCR-VL runner (GPU layout + shared vLLM server, work-stealing).

Usage:
  run.py [pdfs ...] [--workers N] [--vl-conc N] [--server URL]

PDFs to process can be given as:
  - PDF file paths:            run.py a.pdf b.pdf /data/*.pdf
  - directories (*.pdf inside): run.py /data/papers
"""
import os, sys, glob, time, argparse
import multiprocessing as mp

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

def count_pages(p):
    import pypdfium2 as pdfium
    try:
        d = pdfium.PdfDocument(p); n = len(d); d.close(); return n
    except Exception: return 0

def worker(wid, work_q, res_q, barrier, server, vl_conc, output_dir):
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    from paddleocr import PaddleOCRVL
    pipe = PaddleOCRVL(device="gpu:0", vl_rec_backend="vllm-server",
                       vl_rec_server_url=server, vl_rec_max_concurrency=vl_conc,
                       use_queues=True)
    barrier.wait()
    pages = 0
    while True:
        pdf = work_q.get()
        if pdf is None: break
        try:
            pages_res = list(pipe.predict(pdf, use_queues=True))
            pages += len(pages_res)
            out_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(pdf))[0])
            os.makedirs(out_dir, exist_ok=True)
            for res in pipe.restructure_pages(pages_res):
                res.save_to_json(save_path=out_dir)
                res.save_to_markdown(save_path=out_dir)
        except Exception as e:
            res_q.put(("err", wid, str(e)[:150]))
    res_q.put(("done", wid, pages))

def resolve_pdfs(paths):
    pdfs = []
    for p in paths:
        if os.path.isdir(p):
            pdfs += glob.glob(os.path.join(p, "*.pdf"))
        else:
            pdfs.append(p)
    return sorted(set(pdfs))

def main():
    ap = argparse.ArgumentParser(description="Multi-process PaddleOCR-VL runner")
    ap.add_argument("pdfs", nargs="+", help="PDF file paths or directories")
    ap.add_argument("-w", "--workers", type=int, default=8, help="number of worker processes")
    ap.add_argument("-c", "--vl-conc", type=int, default=64, help="vLLM recognition max concurrency per worker")
    ap.add_argument("--server", default="http://localhost:8118/v1", help="vLLM server URL")
    ap.add_argument("-o", "--output-dir", default="output", help="dir to save per-PDF JSON + Markdown")
    args = ap.parse_args()

    pdfs = resolve_pdfs(args.pdfs)
    if not pdfs:
        ap.error("no PDFs found (pass file paths or directories)")

    mp.set_start_method("spawn", force=True)
    os.makedirs(args.output_dir, exist_ok=True)
    total = sum(count_pages(p) for p in pdfs)
    print(f"[cfg] workers={args.workers} vl_conc={args.vl_conc} out={args.output_dir} -> {len(pdfs)} pdfs / {total} pages", flush=True)
    work_q = mp.Queue(); res_q = mp.Queue(); barrier = mp.Barrier(args.workers + 1)
    for p in pdfs: work_q.put(p)
    for _ in range(args.workers): work_q.put(None)
    procs = [mp.Process(target=worker, args=(w, work_q, res_q, barrier, args.server, args.vl_conc, args.output_dir))
             for w in range(args.workers)]
    for pr in procs: pr.start()
    print("[init] loading models...", flush=True)
    barrier.wait(); t0 = time.perf_counter(); print("[run] start", flush=True)
    done = pages = errs = 0
    while done < args.workers:
        tag, wid, pl = res_q.get()
        if tag == "done": done += 1; pages += pl
        else: errs += 1; print(f"[err] w{wid}: {pl}", flush=True)
    dt = time.perf_counter() - t0
    for pr in procs: pr.join()
    print(f"\n==== {args.workers} workers | {pages} pages / {dt:.1f}s = {pages/dt:.3f} pages/sec ({pages/dt*60:.1f} pg/min) | errs={errs} ====", flush=True)

if __name__ == "__main__":
    main()
