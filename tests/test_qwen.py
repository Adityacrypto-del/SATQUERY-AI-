import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    model = model.to(device)

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Qwen2.5-VL-3B loaded successfully.")


if __name__ == "__main__":
    main()