from __future__ import annotations

import argparse
import json

from satquery.single_image_api import analyze_single_image


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery single-image demo")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--query", required=True, help="User query")
    args = parser.parse_args()

    result = analyze_single_image(image_path=args.image, query=args.query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
