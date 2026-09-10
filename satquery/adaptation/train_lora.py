"""LoRA adaptation of Qwen2-VL-7B on RS instruction data.

Uses 4-bit quantization + LoRA to fit in 8-16GB VRAM.

Usage:
    python -m satquery.adaptation.train_lora \\
        --instructions data/rs_instructions/instructions.json \\
        --out checkpoints/rs_vlm_lora \\
        --epochs 1 --batch-size 2 --max-steps 50
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model


class RSInstructionDataset(Dataset):
    def __init__(self, records: list[dict], processor, max_length: int = 256):
        self.records = [r for r in records if r.get("image_path") and Path(r["image_path"]).exists()]
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Optional[dict]:
        rec = self.records[idx]
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            return None

        question = rec.get("question", "Describe this image.")
        answer = rec.get("answer", "")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Answer the following question about this satellite image.\n\nQuestion: {question}\nAnswer:"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        encoding = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        # Labels: encode answer
        answer_ids = self.processor.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        return {
            "pixel_values": encoding.pixel_values.squeeze(0),
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": answer_ids.squeeze(0),
        }


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def train(
    instructions_path: str,
    out_dir: str,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 2e-4,
    max_steps: int = 50,
    token: str = "",
) -> None:
    with open(instructions_path, encoding="utf-8") as f:
        records = json.load(f)

    print(f"[LoRA] {len(records)} training records.")
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"

    use_quantize = device_str == "cuda"
    print(f"[LoRA] Loading {model_id} on {device_str} (quantize={use_quantize}) ...")

    processor = AutoProcessor.from_pretrained(model_id, token=token, trust_remote_code=True)

    if use_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            token=token,
            trust_remote_code=True,
        )
    else:
        dtype = torch.float32 if device_str == "cpu" else torch.float16
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_str if device_str != "mps" else None,
            token=token,
            trust_remote_code=True,
        )
        if device_str == "mps":
            model = model.to(device_str)

    # Configure LoRA on attention layers (q_proj, k_proj, v_proj, o_proj)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = RSInstructionDataset(records, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    step = 0
    for epoch in range(epochs):
        print(f"\n[LoRA] Epoch {epoch + 1}/{epochs}")
        for batch in loader:
            if batch is None:
                continue
            batch = {k: v.to(device_str) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"  step {step:4d} | loss={loss.item():.4f}")
            if step >= max_steps:
                print(f"[LoRA] Reached max_steps={max_steps}. Stopping.")
                break
        if step >= max_steps:
            break

    # Save LoRA adapter
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
    print(f"\n[LoRA] Adapter saved -> {out_path}")

    # Save training summary
    summary = {
        "base_model": model_id,
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "quantization": "4bit-nf4" if use_quantize else "none",
        "num_training_samples": len(dataset),
        "steps": step,
        "epochs": epochs,
        "device": device_str,
        "adapter_path": str(out_path),
    }
    (out_path / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("[LoRA] Training summary saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA adapt Qwen2-VL-7B on RS data")
    parser.add_argument("--instructions", default="data/rs_instructions/instructions.json")
    parser.add_argument("--out", default="checkpoints/rs_vlm_lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    train(
        instructions_path=args.instructions,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        token=args.hf_token,
    )


if __name__ == "__main__":
    main()
