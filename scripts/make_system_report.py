"""Generate the SatQuery AI system report as Markdown and PDF.

One source of truth for both formats. The content lives as structured data
below rather than as prose in two files, so the two cannot drift apart.

Scope is deliberately narrow: what a judge needs to see in a few minutes --
architecture, models, datasets, measured results, and what is honestly not
built yet. Everything here is measured or directly read from the repository;
nothing is estimated.

Usage::

    python -m scripts.make_system_report --out-dir outputs/report
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TITLE = "SatQuery AI - System Report"
SUBTITLE = "SIH 2026 | PS 26167 | ISRO / SAC"

# --------------------------------------------------------------------------
# Content. Each block is (kind, payload).
#   h1/h2/p/note  -> str
#   table         -> {"cols": [...], "widths": [...], "rows": [[...]]}
#   bullets       -> [str, ...]
#   flow          -> [str, ...]   rendered as A -> B -> C
# --------------------------------------------------------------------------

BLOCKS: List[Any] = [
    ("p", "An agentic vision-language assistant for remote-sensing imagery. A controller "
          "reads the query and the input configuration, selects a specialist model, and "
          "returns an evidence-grounded answer with a visual overlay and an auditable "
          "execution trace."),

    ("h1", "1. System architecture"),
    ("flow", ["User query + image(s)", "Controller (task + input routing)",
              "Specialist model", "Evidence + trace", "Web UI"]),
    ("table", {
        "cols": ["Layer", "Technology", "Status"],
        "widths": [46, 88, 36],
        "rows": [
            ["Frontend", "React 19, Vite 8, Tailwind 4 (4 pages)", "Built"],
            ["Backend API", "FastAPI, single-image endpoint", "Built"],
            ["Branch 1: single image", "Qwen2-VL-7B-Instruct + LoRA", "Not adapted"],
            ["Branch 2: bi-temporal", "ResNet-18 U-Net + rule engine", "Measured"],
            ["Branch 3: optical-SAR", "Dual ResNet encoders + Qwen2.5-3B reasoner", "Not trained"],
            ["DeltaVLM adapter", "Interface only (NotImplementedError stubs)", "Not built"],
        ],
    }),
    ("note", "Branches share one contract (ChangeEvidence) so the controller consumes a "
             "single object shape regardless of which specialist answered."),

    ("h1", "2. Branch 1 - single-image VQA and captioning"),
    ("table", {
        "cols": ["Item", "Specification"],
        "widths": [52, 118],
        "rows": [
            ["Base model", "Qwen2-VL-7B-Instruct (vision-language)"],
            ["Adaptation", "LoRA / PEFT, rank 16, alpha 32, q/k/v/o projections"],
            ["Quantisation", "4-bit (8 GB VRAM) or fp16 (16 GB)"],
            ["Input", "GeoTIFF, TIFF, PNG, JPEG via rasterio / PIL"],
            ["Band handling", "RGB by colour interpretation; Sentinel B04/B03/B02 by name;"
                              " otherwise first three bands, recorded in the trace"],
            ["Task routing", "Keyword classifier -> captioning or VQA"],
            ["Outputs", "Answer or caption, model name, execution trace"],
            ["Evaluation", "RSVQA (exact match), VRSBench (BLEU, ROUGE-L, METEOR)"],
            ["Measured accuracy", "NOT YET MEASURED"],
        ],
    }),
    ("note", "The pipeline, LoRA configuration and evaluation scripts are complete, but no "
             "adapter has been trained and no benchmark has been run. The problem statement "
             "requires remote-sensing adaptation explicitly, so this is mandatory scope."),

    ("h1", "3. Branch 2 - bi-temporal change analysis"),
    ("p", "The published CDVQA baseline reaches about 58% because it never uses semantic "
          "labels. This branch predicts what each changed pixel was and became, then applies "
          "the benchmark's rules deterministically."),
    ("table", {
        "cols": ["Item", "Specification"],
        "widths": [52, 118],
        "rows": [
            ["Architecture", "U-Net, two decoders, ResNet-18 ImageNet encoder"],
            ["Parameters", "21.6 M"],
            ["Input / output", "6 channels (2 x RGB) at 512x512 -> two 7-class maps"],
            ["Training", "40 epochs, AMP fp16, RTX 4050 6 GB, best epoch 22"],
            ["Answer engine", "Deterministic rules -> one of CDVQA's 19 answer tokens"],
            ["Inference", "4 s model load, then about 200 ms per query"],
            ["Interface", "One call in, evidence object out; registry descriptor included"],
            ["Tests", "312 passing"],
        ],
    }),
    ("h2", "Measured results"),
    ("table", {
        "cols": ["Metric", "Value", "Note"],
        "widths": [56, 34, 80],
        "rows": [
            ["CDVQA Test average accuracy", "68.14%", "968 scenes, 39,686 questions"],
            ["CDVQA Test overall accuracy", "74.89%", "official held-out split"],
            ["Published CDVQA baseline", "~58%", "we are about 10 points above"],
            ["Majority-class floor", "44.91%", "what guessing alone scores"],
            ["Rule ceiling (oracle)", "99.90%", "rules on ground-truth maps"],
            ["Segmentation mIoU", "42.08%", "diagnostic, not the scored metric"],
        ],
    }),
    ("note", "The oracle figure proves the rules reproduce the benchmark's own answers almost "
             "exactly, so every remaining error is segmentation quality rather than reasoning. "
             "Near-duplicate screening bounds the true test figure at 67.7% or above."),

    ("h1", "4. Branch 3 - optical-SAR paired analysis"),
    ("p", "Optical carries spectral and contextual detail; SAR penetrates cloud and works "
          "day or night. This branch learns a shared representation of the two so a query "
          "can draw on whichever modality actually carries the answer."),
    ("table", {
        "cols": ["Item", "Specification"],
        "widths": [52, 118],
        "rows": [
            ["Optical encoder", "ResNet-18/34, 13-channel first conv (all Sentinel-2 bands, "
                                "not reduced to RGB)"],
            ["SAR encoder", "ResNet-18/34, 2-channel (Sentinel-1 VV + VH backscatter)"],
            ["Embeddings", "512-dim per modality, 256-dim projection heads"],
            ["Alignment", "Symmetric InfoNCE (CLIP-style), temperature 0.07"],
            ["Cloud handling", "Cloud-coverage-aware weighting: heavily clouded optical "
                               "patches contribute less to the alignment loss"],
            ["Fusion", "Concat + MLP (default) or cross-attention, 8 heads"],
            ["Reasoning head", "Qwen2.5-3B-Instruct + LoRA, alpha 32, 64 visual tokens (8x8)"],
            ["Training config", "50 epochs, batch 64, lr 3e-4, 5 warmup epochs"],
            ["API", "FastAPI, /api/optical-sar endpoint"],
            ["Measured accuracy", "NOT YET MEASURED"],
        ],
    }),
    ("note", "The architecture, losses, training loop and API are complete. Only a 3-epoch "
             "smoke run on synthetic data exists; retrieval scores there sit at chance, which "
             "is expected for synthetic input and is not a result. Training on real SEN12MS-CR "
             "data has not been run."),

    ("h1", "5. Datasets"),
    ("table", {
        "cols": ["Dataset", "Role", "Used by", "Scale"],
        "widths": [36, 74, 26, 34],
        "rows": [
            ["SECOND", "Semantic change maps, training labels", "Branch 2", "4,662 pairs"],
            ["CDVQA", "Change VQA benchmark, 19 answers, 8 types", "Branch 2", "2,968 scenes"],
            ["OSCD", "Real Sentinel-2 13-band pairs, GeoTIFF path", "Branch 2", "24 pairs"],
            ["SEN12MS-CR", "Paired Sentinel-1 SAR and Sentinel-2 optical, cloud-affected",
             "Branch 3", "13 + 2 bands"],
            ["BigEarthNet-S2", "Domain adaptation corpus for LoRA", "Branch 1", "subset"],
            ["RSVQA", "Single-image VQA evaluation", "Branch 1", "not yet run"],
            ["VRSBench", "Captioning evaluation", "Branch 1", "not yet run"],
        ],
    }),

    ("h1", "6. Problem statement coverage"),
    ("table", {
        "cols": ["Mandatory requirement", "Status"],
        "widths": [128, 42],
        "rows": [
            ["Single-image VQA", "Built, not measured"],
            ["Second single-image task (captioning)", "Built, not measured"],
            ["Bi-temporal change analysis", "Built and measured"],
            ["Spatial change map", "Built"],
            ["Optical-SAR paired analysis", "Built, not trained"],
            ["Remote-sensing adaptation (fine-tuning)", "NOT DONE"],
            ["Agentic tool orchestration", "Partial"],
            ["Confidence estimation", "Built (branch 2, measured)"],
            ["Visual evidence", "Built (branch 2)"],
            ["Execution trace", "Built (both branches)"],
            ["Downloadable report", "Built (branch 2)"],
            ["Interactive web application", "Built"],
        ],
    }),

    ("h1", "7. What distinguishes this build"),
    ("bullets", [
        "Confidence is measured, never invented. Every number is an accuracy actually "
        "observed on validation for that question type and land-cover class. Where no "
        "measurement applies it reports 0.0 and states why.",
        "Out-of-distribution input is detected. A physics-based producer cross-checks the "
        "trained model; when they disagree the system withdraws its confidence rather than "
        "answering confidently.",
        "Refusals over fabrication. SAR carries no land-cover mapping, so class questions are "
        "refused. Ungeoreferenced input reports pixel counts, never an invented area.",
        "Full execution trace: every stage records the tool, parameters, observation, "
        "duration and why the step was taken.",
        "Visual evidence is built from the same pixels the rule counted, so the picture "
        "cannot contradict the text.",
    ]),

    ("h1", "8. Known gaps"),
    ("bullets", [
        "Remote-sensing fine-tuning on branch 1 has not been run. Mandatory.",
        "Branch 3 optical-SAR is architecturally complete but has never been trained on "
        "real data, so it has no measured capability. Mandatory scope.",
        "Branch 1 has no measured accuracy on any benchmark.",
        "Branch 2 under-predicts rare classes: water at 0.21x true frequency, playgrounds "
        "never predicted. Declared in the tool descriptor and reflected in its confidence.",
        "Branch 2 does not transfer across sensors yet: trained on sub-metre aerial imagery, "
        "it detects nothing at Sentinel-2's 10 m and correctly reports zero confidence.",
        "The controller registry spanning both branches is not yet wired.",
    ]),
]


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------

def to_markdown() -> str:
    out = [f"# {TITLE}", "", f"*{SUBTITLE}*", ""]
    for kind, payload in BLOCKS:
        if kind == "h1":
            out += [f"## {payload}", ""]
        elif kind == "h2":
            out += [f"### {payload}", ""]
        elif kind == "p":
            out += [payload, ""]
        elif kind == "note":
            out += [f"> {payload}", ""]
        elif kind == "flow":
            out += ["`" + "  ->  ".join(payload) + "`", ""]
        elif kind == "bullets":
            out += [f"- {b}" for b in payload] + [""]
        elif kind == "table":
            out.append("| " + " | ".join(payload["cols"]) + " |")
            out.append("|" + "|".join("---" for _ in payload["cols"]) + "|")
            for row in payload["rows"]:
                out.append("| " + " | ".join(str(c) for c in row) + " |")
            out.append("")
    out += ["---", "",
            "All figures are measured on the stated splits. Nothing in this report is "
            "estimated or projected."]
    return "\n".join(out)


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

INK = (16, 24, 32)
MUTED = (101, 121, 138)
ACCENT = (185, 82, 10)
RULE = (201, 212, 220)
BAND = (238, 243, 246)


def to_pdf(path: str) -> str:
    from fpdf import FPDF

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(16, 15, 16)
    pdf.add_page()

    def rule(colour=RULE, thickness=0.3):
        pdf.set_draw_color(*colour)
        pdf.set_line_width(thickness)
        y = pdf.get_y()
        pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
        pdf.ln(2)

    # masthead
    pdf.set_text_color(*ACCENT)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(0, 4, SUBTITLE.upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    pdf.set_text_color(*INK)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 9, TITLE, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)
    rule(ACCENT, 0.8)
    pdf.ln(2)

    for kind, payload in BLOCKS:
        if kind == "h1":
            if pdf.get_y() > 240:
                pdf.add_page()
            pdf.ln(3)
            pdf.set_text_color(*INK)
            pdf.set_font("Helvetica", "B", 12.5)
            pdf.cell(0, 6, payload, new_x="LMARGIN", new_y="NEXT")
            rule()
            pdf.ln(1)

        elif kind == "h2":
            pdf.ln(1.5)
            pdf.set_text_color(*INK)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 5, payload, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)

        elif kind == "p":
            pdf.set_text_color(*INK)
            pdf.set_font("Helvetica", "", 9.2)
            pdf.multi_cell(0, 4.4, payload, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)

        elif kind == "note":
            pdf.set_fill_color(*BAND)
            pdf.set_text_color(*MUTED)
            pdf.set_font("Helvetica", "I", 8.6)
            pdf.multi_cell(0, 4.2, payload, new_x="LMARGIN", new_y="NEXT",
                           fill=True, padding=2)
            pdf.ln(2)

        elif kind == "flow":
            pdf.set_text_color(*ACCENT)
            pdf.set_font("Helvetica", "B", 8.6)
            pdf.multi_cell(0, 4.6, "   >   ".join(payload),
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

        elif kind == "bullets":
            pdf.set_text_color(*INK)
            for item in payload:
                pdf.set_font("Helvetica", "B", 9.2)
                pdf.cell(4, 4.4, "-")
                pdf.set_font("Helvetica", "", 9.2)
                pdf.multi_cell(0, 4.4, item, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(0.8)
            pdf.ln(1)

        elif kind == "table":
            cols, widths, rows = payload["cols"], payload["widths"], payload["rows"]
            if pdf.get_y() + 10 + 6 * len(rows) > 275:
                pdf.add_page()
            pdf.set_fill_color(*BAND)
            pdf.set_text_color(*MUTED)
            pdf.set_font("Helvetica", "B", 7.8)
            for col, width in zip(cols, widths):
                pdf.cell(width, 5.6, " " + col.upper(), fill=True, border=0)
            pdf.ln(5.6)
            pdf.set_text_color(*INK)
            for index, row in enumerate(rows):
                # Height is set by the tallest cell, so wrapped text cannot
                # overlap the row beneath it.
                pdf.set_font("Helvetica", "", 8.4)
                lines = max(
                    len(pdf.multi_cell(width - 2, 4, str(cell), dry_run=True,
                                       output="LINES"))
                    for cell, width in zip(row, widths)
                )
                height = max(5.2, lines * 4 + 1.2)
                if pdf.get_y() + height > 280:
                    pdf.add_page()
                top, left = pdf.get_y(), pdf.l_margin
                for column, (cell, width) in enumerate(zip(row, widths)):
                    pdf.set_xy(left, top)
                    bold = column == 0
                    pdf.set_font("Helvetica", "B" if bold else "", 8.4)
                    flag = str(cell).upper()
                    if flag.startswith("NOT ") or flag == "PARTIAL":
                        pdf.set_text_color(*ACCENT)
                    pdf.multi_cell(width, 4, " " + str(cell), align="L",
                                   max_line_height=4)
                    pdf.set_text_color(*INK)
                    left += width
                pdf.set_xy(pdf.l_margin, top + height)
                pdf.set_draw_color(*RULE)
                pdf.set_line_width(0.15)
                pdf.line(pdf.l_margin, pdf.get_y() - 0.6,
                         pdf.w - pdf.r_margin, pdf.get_y() - 0.6)
            pdf.ln(3)

    pdf.ln(2)
    rule()
    pdf.set_text_color(*MUTED)
    pdf.set_font("Helvetica", "I", 7.6)
    pdf.multi_cell(0, 3.6,
                   "All figures are measured on the stated splits. "
                   "Nothing in this report is estimated or projected.",
                   new_x="LMARGIN", new_y="NEXT")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pdf.output(path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="outputs/report")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "SatQuery_System_Report.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(to_markdown())
    print(f"markdown -> {md_path}")

    try:
        pdf_path = to_pdf(os.path.join(args.out_dir, "SatQuery_System_Report.pdf"))
        print(f"pdf      -> {pdf_path}")
    except Exception as exc:  # noqa: BLE001 - markdown is the guaranteed output
        print(f"pdf      -> skipped ({type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
