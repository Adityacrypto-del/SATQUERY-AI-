"""Score a trained checkpoint on a named CDVQA split.

A thin harness. All the scoring logic lives in the already-tested
``CDVQAValidator``; this only loads a checkpoint, builds a loader over the
split's scenes in validator order, and writes the report out.

Test and Test2 are budgeted (see ``outputs/eval_log.md``): Test at most three
times in the project, Test2 exactly once at the very end. Running either
requires ``--i-am-spending-test-budget`` together with a written reason, so it
cannot happen by reflex or by a mistyped ``--split``. The reason is recorded
in the report next to the number it bought.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.dataset import SECONDChangeDataset
from segmentation.train import SharedChangeNet, SiameseChangeNet
from segmentation.validate_cdvqa import CDVQAValidator, format_validation

BUDGETED_SPLITS = ("Test", "Test2")


def build_model(payload) -> torch.nn.Module:
    """Architecture is read from the weights, never from a default.

    Run 1 and Run 2 predate the ``arch`` config field, so a default would
    pick the architecture by luck. Only the shared model has a change head,
    and only a Run 5 checkpoint has a seventh class-head channel.
    """
    state = payload["model"]
    inferred = "shared" if any(k.startswith("change_head") for k in state) else "siamese"
    declared = (payload.get("config") or {}).get("arch")
    if declared and declared != inferred:
        raise ValueError(
            f"checkpoint declares arch={declared!r} but its weights look like "
            f"{inferred!r}; refusing to score a mismatched model"
        )
    if inferred == "shared":
        width = int(state["class_heads.0.weight"].shape[0])
        model = SharedChangeNet(pretrained=False, unchanged_logit=width > 6)
    else:
        model = SiameseChangeNet(pretrained=False)
    model.load_state_dict(state)
    return model, inferred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="Val",
                        choices=("Train", "Val", "Test", "Test2"))
    parser.add_argument("--cdvqa-root", default="datasets/CDVQA")
    parser.add_argument("--second-root", default="datasets/SECOND_raw/train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--out", default=None, help="write the report JSON here")
    parser.add_argument("--i-am-spending-test-budget", action="store_true",
                        help="required for Test/Test2; see outputs/eval_log.md")
    parser.add_argument("--reason", default="",
                        help="why this budgeted evaluation is being spent")
    args = parser.parse_args(argv)

    if args.split in BUDGETED_SPLITS:
        if not args.i_am_spending_test_budget:
            parser.error(
                f"{args.split} is budgeted. Pass --i-am-spending-test-budget "
                f"and --reason, and append the result to outputs/eval_log.md."
            )
        if not args.reason.strip():
            parser.error("a budgeted evaluation requires --reason")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, arch = build_model(payload)
    model = model.to(device).eval()

    validator = CDVQAValidator(args.cdvqa_root, args.second_root, split=args.split)
    dataset = SECONDChangeDataset(args.second_root, validator.scenes, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"checkpoint : {args.checkpoint}")
    print(f"arch       : {arch} (inferred from weights)")
    print(f"epoch      : {payload.get('epoch')}")
    print(f"split      : {args.split}  ({len(validator.scenes)} scenes)")
    if args.reason:
        print(f"reason     : {args.reason}")
    print()

    report = validator.run(model, loader, device, amp=not args.no_amp)
    print(format_validation(report))

    report["evaluated_at"] = datetime.now().isoformat(timespec="seconds")
    report["checkpoint"] = args.checkpoint
    report["arch"] = arch
    report["checkpoint_epoch"] = payload.get("epoch")
    report["reason"] = args.reason or None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
