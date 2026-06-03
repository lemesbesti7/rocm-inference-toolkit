#!/usr/bin/env python3
"""Example: quantize a model for faster ROCm inference."""

from quantize import quantize_model


def main():
    # Quantize BERT-base to INT8
    quantize_model(
        model_name="bert-base-uncased",
        output_dir="./quantized_bert_int8",
        method="dynamic",
    )

    # Quantize to FP16
    quantize_model(
        model_name="bert-base-uncased",
        output_dir="./quantized_bert_fp16",
        method="fp16",
    )

    print("\nQuantized models saved. Use with:")
    print("  python inference.py --model ./quantized_bert_int8 --precision int8")


if __name__ == "__main__":
    main()
