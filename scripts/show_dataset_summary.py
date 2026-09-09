from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    summary_path = Path("data/download_summary.json")
    if not summary_path.exists():
        print("No summary found. Run scripts/download_datasets.py first.")
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))

    manifest = Path(summary["vrsbench"]["manifest"])
    if manifest.exists():
        items = json.loads(manifest.read_text(encoding="utf-8"))
        print("\nSample rows:")
        for row in items[:5]:
            qa_pairs = row.get("qa_pairs") or []
            first_qa = qa_pairs[0] if qa_pairs else {}
            print(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "image_path": row.get("image_path"),
                        "image_name": row.get("image_name"),
                        "caption": row.get("caption"),
                        "sample_question": first_qa.get("question"),
                        "sample_answer": first_qa.get("answer"),
                    },
                    ensure_ascii=True,
                )
            )


if __name__ == "__main__":
    main()
