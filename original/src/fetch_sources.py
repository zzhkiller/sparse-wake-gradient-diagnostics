"""Fetch only public research metadata and, optionally, the author's small LES subset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
HEADERS = {"User-Agent": "Wake-Reconstruction-Research/1.0", "Accept": "application/json"}
AUTHOR_COMMIT = "dda521625a96dce48f138f53b512fdf82db0a266"
URLS = {
    "dtu_datacite": "https://api.datacite.org/dois/10.11583/DTU.10259936.v1",
    "author_repository": f"https://api.github.com/repos/fengguoFUAS/PINN-Lidar-WFR/contents?ref={AUTHOR_COMMIT}",
    "author_commit": f"https://api.github.com/repos/fengguoFUAS/PINN-Lidar-WFR/commits/{AUTHOR_COMMIT}",
    "rom_wfe_crossref": "https://api.crossref.org/works/10.1016/j.energy.2024.130772",
    "enginebench_crossref": "https://api.crossref.org/works/10.1177/14680874251330354",
    "wake_experiment_crossref": "https://api.crossref.org/works/10.1126/sciadv.adw8524",
    "pinn_pod_crossref": "https://api.crossref.org/works/10.1063/5.0292127",
}


def fetch(item):
    name, url = item
    record = {"name": name, "url": url, "checked_utc": datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=35) as response:
            body = response.read(8_000_001)
            if len(body) > 8_000_000:
                raise ValueError("Metadata exceeds the explicit 8 MB limit")
            obj = json.loads(body)
            (SOURCES / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2))
            record.update(status=response.status, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    return record


def download_author_data():
    entries = json.loads((SOURCES / "author_repository.json").read_text())
    commit = json.loads((SOURCES / "author_commit.json").read_text())["sha"]
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    records = []
    for entry in entries:
        if not entry["name"].endswith(".mat"):
            continue
        if entry["size"] > 120_000_000:
            records.append({"name": entry["name"], "status": "skipped_size", "bytes": entry["size"]})
            continue
        url = f"https://raw.githubusercontent.com/fengguoFUAS/PINN-Lidar-WFR/{commit}/{entry['name']}"
        path = data_dir / entry["name"]
        digest = hashlib.sha256()
        git_digest = hashlib.sha1(f"blob {entry['size']}\0".encode())
        if not path.exists():
            tmp = path.with_suffix(".mat.partial")
            try:
                with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as stream:
                    total = 0
                    while block := response.read(1_048_576):
                        total += len(block)
                        if total > 120_000_000:
                            raise ValueError("Download exceeded 120 MB per-file limit")
                        stream.write(block)
                if total != entry["size"]:
                    raise ValueError(f"Size mismatch: received {total}, expected {entry['size']}")
                tmp.rename(path)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        with path.open("rb") as stream:
            while block := stream.read(1_048_576):
                digest.update(block)
                git_digest.update(block)
        if git_digest.hexdigest() != entry["sha"]:
            raise ValueError(f"Git blob checksum mismatch: {entry['name']}")
        records.append({"name": entry["name"], "url": url, "commit": commit,
                        "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                        "git_blob_sha1":git_digest.hexdigest(), "status": "downloaded"})
        print(json.dumps(records[-1]), flush=True)
    (SOURCES / "data_download_manifest.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-author-data", action="store_true")
    args = parser.parse_args()
    SOURCES.mkdir(exist_ok=True)
    records = list(ThreadPoolExecutor(4).map(fetch, URLS.items()))
    (SOURCES / "fetch_log.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)
    if args.download_author_data:
        download_author_data()


if __name__ == "__main__":
    main()
