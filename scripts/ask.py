"""Ask the branch-2 specialist about an image pair, from the terminal.

One-shot::

    python -m scripts.ask --t1 a.tif --t2 b.tif -q "Have buildings increased?"

Interactive, so a pair can be interrogated from several angles without
paying the checkpoint load each time::

    python -m scripts.ask --t1 a.tif --t2 b.tif

This is a thin front end over ``BiTemporalSpecialist`` -- the same call the
controller makes, with nothing added. What it prints is what the controller
receives, so if the answer looks wrong here it is wrong there too.

``--trace`` prints the execution trace, which is the artefact the problem
statement actually grades: every stage with the tool that ran, its
parameters, what it observed and why the step was taken.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CHECKPOINT = "outputs/segmentation/run1_unweighted/best.pt"

SUGGESTIONS = [
    "What changed and where?",
    "Have the areas of buildings changed?",
    "Did the areas of low vegetation increase?",
    "What have the regions of low vegetation mainly changed to?",
    "What is the percentage of changed areas?",
    "What did low vegetation and water change into?",
]


def _rule(char: str = "-", width: int = 72) -> str:
    return char * width


def render(evidence, show_trace: bool = False) -> None:
    print()
    print(_rule("="))
    if evidence.answer is not None:
        print(f"  ANSWER      {evidence.answer}")
    elif getattr(evidence, "compound_answers", None):
        print("  ANSWER      (per class, no single benchmark token applies)")
        for name, value in evidence.compound_answers.items():
            shown = value if value is not None else "no answer"
            print(f"                {name.replace('_', ' ')}: {shown}")
    else:
        print("  ANSWER      (none -- no rule applies; description below)")

    print(f"  CONFIDENCE  {evidence.confidence:.4f}")
    # The basis is the point. A bare 0.0 cannot be told from "very unsure",
    # and a bare 0.94 cannot be told from "right for the wrong reason".
    for line in _wrap(evidence.confidence_basis, 58):
        print(f"              {line}")

    print(_rule())
    print(f"  question type  {evidence.question_type}")
    print(f"  route          {evidence.route}")
    print(f"  changed        {evidence.changed}")
    pixels = evidence.changed_area_pixels
    area = evidence.changed_area_m2
    print(f"  changed area   {pixels:,} px" if pixels is not None else
          "  changed area   not computed")
    if area is not None:
        print(f"                 {area:,.0f} m2")
    else:
        print("                 area in m2 unavailable (input not georeferenced)")
    print(f"  regions        {len(evidence.regions)}")

    print(_rule())
    print("  SUMMARY")
    for line in _wrap(evidence.summary, 66):
        print(f"    {line}")

    if evidence.overlays:
        print(_rule())
        print("  VISUAL EVIDENCE")
        for name, path in evidence.overlays.items():
            print(f"    {name:<18} {path}")

    if show_trace:
        print(_rule())
        print(f"  EXECUTION TRACE ({len(evidence.trace)} stages)")
        for entry in evidence.trace:
            print(f"    [{entry['stage']}] {entry['name']}  "
                  f"({entry['duration_ms']:.1f} ms)")
            print(f"        tool: {entry['tool']}")
            print(f"        why : {entry['why']}")
            for line in _wrap(entry["observation"], 58):
                print(f"        ->   {line}")
    print(_rule("="))


def _wrap(text: str, width: int):
    import textwrap

    return textwrap.wrap(str(text), width) or [""]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", required=True, help="earlier image")
    parser.add_argument("--t2", required=True, help="later image")
    parser.add_argument("-q", "--query", default=None,
                        help="one question; omit for an interactive session")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--overlay-dir", default="outputs/evidence")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trace", action="store_true",
                        help="print the execution trace")
    parser.add_argument("--describe", action="store_true",
                        help="print the tool-registry descriptor and exit")
    args = parser.parse_args(argv)

    from tools.change_analysis.api import BiTemporalSpecialist, describe_tool

    if args.describe:
        import json

        print(json.dumps(describe_tool(args.checkpoint), indent=2))
        return 0

    for path in (args.t1, args.t2):
        if not os.path.exists(path):
            parser.error(f"no such file: {path}")

    checkpoint = args.checkpoint if os.path.exists(args.checkpoint) else None
    if checkpoint is None:
        print(f"note: no checkpoint at {args.checkpoint!r}; running the "
              f"deterministic index path only.", file=sys.stderr)

    print(f"loading {checkpoint or 'index path (no checkpoint)'} ...",
          file=sys.stderr, flush=True)
    specialist = BiTemporalSpecialist(
        checkpoint=checkpoint, overlay_dir=args.overlay_dir, device=args.device
    )
    stem = os.path.splitext(os.path.basename(args.t1))[0]

    if args.query is not None:
        render(specialist.analyze(args.t1, args.t2, args.query, scene_id=stem),
               show_trace=args.trace)
        return 0

    print()
    print(f"  t1: {args.t1}")
    print(f"  t2: {args.t2}")
    print("\n  Try one of these, or type your own:")
    for suggestion in SUGGESTIONS:
        print(f"    - {suggestion}")
    print("\n  Commands: 'trace' toggles the execution trace, "
          "'describe' prints the\n            tool descriptor, "
          "'quit' exits.\n")

    show_trace = args.trace
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            return 0
        if query.lower() == "trace":
            show_trace = not show_trace
            print(f"  execution trace {'on' if show_trace else 'off'}")
            continue
        if query.lower() == "describe":
            import json

            print(json.dumps(describe_tool(checkpoint), indent=2))
            continue
        try:
            render(specialist.analyze(args.t1, args.t2, query, scene_id=stem),
                   show_trace=show_trace)
        except Exception as exc:  # noqa: BLE001 - a REPL must survive one bad query
            print(f"  failed: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
