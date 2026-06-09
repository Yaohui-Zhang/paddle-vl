#!/usr/bin/env python3
"""Fetch ~200 random arXiv papers as PDFs into /workspace/papers.

Random = spread queries over many categories with random start offsets,
then shuffle the collected ids and download the first 200 that succeed.
Polite: a few seconds between API calls, retries on download failure.
"""
import os, time, random, urllib.request, urllib.parse, urllib.error, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

OUT = "/workspace/papers"
TARGET = 200
API = "https://export.arxiv.org/api/query"
CATS = [
    "cs.CL", "cs.CV", "cs.LG", "cs.AI", "cs.RO", "cs.CR", "cs.DC", "cs.SE",
    "math.PR", "math.OC", "math.NA", "stat.ML", "eess.SP", "eess.IV",
    "physics.optics", "cond-mat.stat-mech", "astro-ph.GA", "q-bio.NC",
    "econ.EM", "quant-ph",
]
HDRS = {"User-Agent": "Mozilla/5.0 (paper-fetch; mailto:yaohui@gxl.ai)"}
random.seed()  # nondeterministic

os.makedirs(OUT, exist_ok=True)


def fetch(url):
    req = urllib.request.Request(url, headers=HDRS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def collect_ids():
    ids = set()
    random.shuffle(CATS)
    for cat in CATS:
        if len(ids) >= TARGET * 2:
            break
        start = random.randint(0, 5000)
        q = urllib.parse.urlencode({
            "search_query": f"cat:{cat}",
            "start": start,
            "max_results": 50,
            "sortBy": "lastUpdatedDate",
            "sortOrder": "descending",
        })
        try:
            xml = fetch(f"{API}?{q}").decode("utf-8", "replace")
        except Exception as e:
            print(f"[api] {cat} start={start} failed: {e}", flush=True)
            time.sleep(3)
            continue
        found = re.findall(r"<id>https?://arxiv\.org/abs/([^<]+)</id>", xml)
        ids.update(found)
        print(f"[api] {cat} start={start}: +{len(found)} (total {len(ids)})", flush=True)
        time.sleep(3)  # be polite to the API
    return list(ids)


WORKERS = 16
_lock = threading.Lock()
_count = {"ok": 0}


def download(aid):
    # strip version suffix for the filename, keep it for the URL
    safe = aid.replace("/", "_")
    path = os.path.join(OUT, f"{safe}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return True
    url = f"https://arxiv.org/pdf/{aid}"
    try:
        data = fetch(url)
    except Exception as e:
        print(f"[pdf] {aid} failed: {e}", flush=True)
        return False
    if not data[:4] == b"%PDF":
        print(f"[pdf] {aid} not a pdf (got {data[:20]!r})", flush=True)
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def main():
    ids = collect_ids()
    random.shuffle(ids)
    have = len([f for f in os.listdir(OUT) if f.endswith(".pdf")])
    print(f"[plan] collected {len(ids)} ids, {have} already present, "
          f"downloading with {WORKERS} threads until {TARGET} succeed", flush=True)

    def task(aid):
        if _count["ok"] >= TARGET:
            return False
        ok = download(aid)
        if ok:
            with _lock:
                _count["ok"] += 1
                n = _count["ok"]
                if n % 10 == 0:
                    print(f"[dl] {n}/{TARGET}", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(task, aid) for aid in ids]
        for _ in as_completed(futs):
            if _count["ok"] >= TARGET:
                break

    final = len([f for f in os.listdir(OUT) if f.endswith(".pdf")])
    print(f"[done] {final} pdfs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
