#!/usr/bin/env python3
"""Clean multi-process scaling test (GPU layout + shared vLLM server, work-stealing).
Usage: bench_mp.py [n_pdfs] [n_workers] [vl_conc] [offset]
"""
import os, sys, glob, time
import multiprocessing as mp

N_PDFS  = int(sys.argv[1]) if len(sys.argv) > 1 else 10
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
VL_CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 64
OFFSET  = int(sys.argv[4]) if len(sys.argv) > 4 else 80
SERVER  = "http://localhost:8118/v1"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

def count_pages(p):
    import pypdfium2 as pdfium
    try:
        d = pdfium.PdfDocument(p); n = len(d); d.close(); return n
    except Exception: return 0

def worker(wid, work_q, res_q, barrier):
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    from paddleocr import PaddleOCRVL
    pipe = PaddleOCRVL(device="gpu:0", vl_rec_backend="vllm-server",
                       vl_rec_server_url=SERVER, vl_rec_max_concurrency=VL_CONC,
                       use_queues=True)
    barrier.wait()
    pages = 0
    while True:
        pdf = work_q.get()
        if pdf is None: break
        try:
            for _ in pipe.predict(pdf, use_queues=True): pages += 1
        except Exception as e:
            res_q.put(("err", wid, str(e)[:150]))
    res_q.put(("done", wid, pages))

def main():
    mp.set_start_method("spawn", force=True)
    pdfs = sorted(glob.glob("/workspace/papers/*.pdf"))[OFFSET:OFFSET + N_PDFS]
    total = sum(count_pages(p) for p in pdfs)
    print(f"[cfg] workers={WORKERS} vl_conc={VL_CONC} offset={OFFSET} -> {len(pdfs)} pdfs / {total} pages", flush=True)
    work_q = mp.Queue(); res_q = mp.Queue(); barrier = mp.Barrier(WORKERS + 1)
    for p in pdfs: work_q.put(p)
    for _ in range(WORKERS): work_q.put(None)
    procs = [mp.Process(target=worker, args=(w, work_q, res_q, barrier)) for w in range(WORKERS)]
    for pr in procs: pr.start()
    print("[init] loading models...", flush=True)
    barrier.wait(); t0 = time.perf_counter(); print("[run] start", flush=True)
    done = pages = errs = 0
    while done < WORKERS:
        tag, wid, pl = res_q.get()
        if tag == "done": done += 1; pages += pl
        else: errs += 1; print(f"[err] w{wid}: {pl}", flush=True)
    dt = time.perf_counter() - t0
    for pr in procs: pr.join()
    print(f"\n==== {WORKERS} workers | {pages} pages / {dt:.1f}s = {pages/dt:.3f} pages/sec ({pages/dt*60:.1f} pg/min) | errs={errs} ====", flush=True)

if __name__ == "__main__":
    main()