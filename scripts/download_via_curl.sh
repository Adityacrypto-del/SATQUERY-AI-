#!/usr/bin/env bash
# download_via_curl.sh — Downloads VRSBench samples using curl + HF API directly
# Usage: bash scripts/download_via_curl.sh <HF_TOKEN> <N_SAMPLES>

set -euo pipefail

TOKEN="${1:-}"
N="${2:-20}"
OUT="data/vrsbench/sample"
IMAGES_DIR="$OUT/images"
REPO="xiang709/VRSBench"

if [ -z "$TOKEN" ]; then
  echo "Error: HF token required as first argument"
  exit 1
fi

mkdir -p "$IMAGES_DIR"
echo "[curl-dl] Fetching VRSBench dataset info from HuggingFace..."

# Get the parquet file list for the train split
PARQUET_URL="https://datasets-server.huggingface.co/parquet?dataset=${REPO}&config=default&split=train"
PARQUET_INFO=$(curl -s -H "Authorization: Bearer $TOKEN" "$PARQUET_URL")
echo "$PARQUET_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); urls=[p['url'] for p in d.get('parquet_files',[])]; print('\n'.join(urls[:3]))" > /tmp/parquet_urls.txt 2>/dev/null || true

FIRST_URL=$(head -1 /tmp/parquet_urls.txt)
if [ -z "$FIRST_URL" ]; then
  echo "[curl-dl] Could not get parquet URL, trying direct dataset API..."
  FIRST_URL="https://huggingface.co/datasets/${REPO}/resolve/main/data/train-00000-of-00001.parquet"
fi

echo "[curl-dl] Downloading parquet from: $FIRST_URL"
curl -L -s -H "Authorization: Bearer $TOKEN" "$FIRST_URL" -o /tmp/vrsbench_shard.parquet
echo "[curl-dl] Parquet downloaded ($(du -sh /tmp/vrsbench_shard.parquet | cut -f1))."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
  VENV_PYTHON="python3"
fi

echo "[curl-dl] Extracting $N images from parquet..."
"$VENV_PYTHON" - "$IMAGES_DIR" "$N" "$OUT" <<'PYEOF'
import sys, io, json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

images_dir = Path(sys.argv[1])
n = int(sys.argv[2])
out = Path(sys.argv[3])
shard = "/tmp/vrsbench_shard.parquet"

print(f"[extract] Reading {shard}...")
table = pq.read_table(shard)
print(f"[extract] Columns: {table.column_names}")
print(f"[extract] Total rows: {len(table)}")

records = []
for i, row in enumerate(table.to_pydict().get("image", [])[:n]):
    if isinstance(row, dict) and "bytes" in row:
        img_bytes = row["bytes"]
        if img_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                p = images_dir / f"vrsbench_{i:04d}.jpg"
                img.save(str(p), format="JPEG", quality=90)
                records.append({"id": i, "image_path": str(p)})
                print(f"  [{i+1}/{n}] saved {p.name}")
            except Exception as e:
                print(f"  [{i+1}] error: {e}")

captions = table.to_pydict().get("caption", table.to_pydict().get("Captions", [None] * n))
qa_list = table.to_pydict().get("qa_pairs", table.to_pydict().get("Questions", [None] * n))

for i, rec in enumerate(records):
    rec["caption"] = captions[i] if i < len(captions) else ""
    rec["qa_pairs"] = qa_list[i] if i < len(qa_list) else []
    if isinstance(rec["caption"], list):
        rec["caption"] = " ".join(rec["caption"])

manifest = out / "manifest.json"
manifest.write_text(json.dumps(records, indent=2))
print(f"\n[extract] Saved {len(records)} records to {manifest}")
PYEOF

echo "[curl-dl] Done. Check data/vrsbench/sample/manifest.json"
