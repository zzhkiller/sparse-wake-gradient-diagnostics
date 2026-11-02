"""Download the pinned public LES excerpt, verify hashes, and copy to both runs."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "sources/data_download_manifest.json").read_text())
for subdir in ("original", "revision"):
    (ROOT / subdir / "data").mkdir(parents=True, exist_ok=True)
for item in manifest:
    path = ROOT / "original/data" / item["name"]
    if not path.exists():
        temporary = path.with_suffix(".mat.partial")
        try:
            with urllib.request.urlopen(item["url"], timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if temporary.stat().st_size != item["bytes"] or hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded data failed size/hash check: {item['name']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.stat().st_size != item["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit(f"Existing data failed size/hash check: {path}")
    shutil.copy2(path, ROOT / "revision/data" / item["name"])
    print(f"Verified and installed {item['name']}")
