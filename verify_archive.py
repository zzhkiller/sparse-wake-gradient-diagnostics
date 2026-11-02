"""Verify archived file integrity without downloading data or running training."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "SHA256.json").read_text())
for name, expected in manifest.items():
    path = ROOT / name
    if not path.is_file():
        raise SystemExit(f"Missing: {name}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA256 mismatch: {name}")
print(f"PASS: {len(manifest)} archived files match SHA256.json")
