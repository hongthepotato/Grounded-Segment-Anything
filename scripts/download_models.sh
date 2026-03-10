#!/usr/bin/env bash

set -euo pipefail

MANIFEST_PATH="${MODEL_MANIFEST_PATH:-configs/model_urls.json}"
MODEL_ROOT="${MODEL_ROOT_DIR:-data/models/pretrained}"

mkdir -p "${MODEL_ROOT}"

if [ ! -f "${MANIFEST_PATH}" ]; then
    echo "Model manifest not found at ${MANIFEST_PATH}; skipping model bootstrap."
    exit 0
fi

if [ ! -s "${MANIFEST_PATH}" ]; then
    echo "Model manifest is empty at ${MANIFEST_PATH}; skipping model bootstrap."
    exit 0
fi

echo "Using model manifest: ${MANIFEST_PATH}"
echo "Model root: ${MODEL_ROOT}"

python - "$MANIFEST_PATH" "$MODEL_ROOT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import urllib.request

manifest_path = pathlib.Path(sys.argv[1])
model_root = pathlib.Path(sys.argv[2])

with manifest_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

models = data.get("models", [])
if not isinstance(models, list):
    raise SystemExit("Invalid manifest: 'models' must be a list")

if not models:
    print("No models listed in manifest; skipping downloads.")
    raise SystemExit(0)

def sha256sum(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for entry in models:
    if not isinstance(entry, dict):
        raise SystemExit("Invalid model entry: each item must be an object")

    name = entry.get("name")
    url = entry.get("url")
    expected_sha = entry.get("sha256")

    if not name or not url:
        raise SystemExit("Invalid model entry: 'name' and 'url' are required")

    target_path = model_root / name
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        if expected_sha:
            actual = sha256sum(target_path)
            if actual.lower() == expected_sha.lower():
                print(f"[ok] {name} already exists and checksum matches")
                continue
            print(f"[warn] {name} exists but checksum mismatch, re-downloading")
        else:
            print(f"[ok] {name} already exists, skipping")
            continue

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    print(f"[dl] {name} <- {url}")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, target_path)

    if expected_sha:
        actual = sha256sum(target_path)
        if actual.lower() != expected_sha.lower():
            raise SystemExit(
                f"Checksum mismatch for {name}: expected {expected_sha}, got {actual}"
            )
        print(f"[ok] {name} checksum verified")
    else:
        print(f"[ok] {name} downloaded (no checksum configured)")
PY
