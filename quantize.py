"""Model quantization utilities for ROCm."""

import torch
from transformers import AutoModel, AutoTokenizer


def quantize_model(model_name: str, output_dir: str, method: str = "dynamic"):
    """Quantize a model for faster inference on ROCm."""
    print(f"Quantizing {model_name} with method={method}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)

    if method == "dynamic":
        quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8
        )
    elif method == "fp16":
        quantized = model.half()
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    quantized.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved to {output_dir}")
    return quantized


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="./quantized_model")
    parser.add_argument("--method", default="dynamic", choices=["dynamic", "fp16"])
    args = parser.parse_args()
    quantize_model(args.model, args.output, args.method)
