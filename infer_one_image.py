#!/usr/bin/env python3
"""
Load a pretrained trustworthy GC ImageNet model and run inference on one image.

Example:
  python infer_one_image.py path/to/image.jpg
  python infer_one_image.py path/to/image.jpg --beta inf --weights local.pt

Requires: CUDA (see project README), FrEIA 0.2, dependencies from requirements.txt.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ibinn_imagenet.model.classifiers.invertible_imagenet_classifier import (
    trustworthy_gc_beta_0,
    trustworthy_gc_beta_1,
    trustworthy_gc_beta_2,
    trustworthy_gc_beta_4,
    trustworthy_gc_beta_8,
    trustworthy_gc_beta_16,
    trustworthy_gc_beta_32,
    trustworthy_gc_beta_inf,
)

IMAGENET_LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
)

BETA_LOADERS = {
    "0": trustworthy_gc_beta_0,
    "1": trustworthy_gc_beta_1,
    "2": trustworthy_gc_beta_2,
    "4": trustworthy_gc_beta_4,
    "8": trustworthy_gc_beta_8,
    "16": trustworthy_gc_beta_16,
    "32": trustworthy_gc_beta_32,
    "inf": trustworthy_gc_beta_inf,
}


def print_cuda_help() -> None:
    print(
        """
CUDA 不可用时的处理办法:

  情况 A — 本机有 NVIDIA 独立显卡
    1) 安装或更新显卡驱动，在终端运行 nvidia-smi 应能显示 GPU。
    2) 安装「带 CUDA」的 PyTorch（pip 默认常常是 CPU 版，会导致本错误）。
       打开 https://pytorch.org 按你的 CUDA 版本选择命令，例如 CUDA 12.4:
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
       （具体 cu12x 以官网为准）
    3) 验证: python -c "import torch; print(torch.cuda.is_available())"
       应输出 True。

  情况 B — 没有 NVIDIA GPU（或只有核显）
    本仓库里 DCT 等模块在代码中要求 CUDA，当前无法在纯 CPU 上跑通该推理脚本。
    可使用: Google Colab 免费 GPU、学校实验室机器、云 GPU 等，在对应环境中安装依赖后再运行。

路径提示: 第一个参数必须是「单张图片」文件路径（如 .jpg），不能是文件夹。
""",
        file=sys.stderr,
    )


def load_imagenet_class_names() -> list[str]:
    try:
        with urllib.request.urlopen(IMAGENET_LABELS_URL, timeout=15) as r:
            text = r.read().decode("utf-8")
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if len(lines) == 1000:
            return lines
    except Exception as e:
        print(f"Warning: could not download ImageNet class names ({e}). Showing indices only.")
    return []


def build_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Single-image inference with pretrained trustworthy GC model.")
    p.add_argument("image", help="Path to an RGB image (jpg/png/...)")
    p.add_argument(
        "--beta",
        default="0",
        choices=sorted(BETA_LOADERS.keys()),
        help="Which pretrained variant to load (default: 0).",
    )
    p.add_argument(
        "--weights",
        default=None,
        metavar="PATH",
        help="Optional path to a local .pt checkpoint (.avg.pt). If omitted, weights are downloaded.",
    )
    p.add_argument("--topk", type=int, default=5, help="How many top classes to print.")
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(
            f"Error: not a single image file: {args.image}\n"
            "       Pass one image path (e.g. folder\\\\photo.jpg), not a directory.",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print(
            "Error: CUDA is not available. This model requires an NVIDIA GPU + CUDA-enabled PyTorch.",
            file=sys.stderr,
        )
        print_cuda_help()
        return 1

    beta_key = args.beta
    loader = BETA_LOADERS[beta_key]

    print(f"Loading pretrained model (beta={beta_key})...")
    model = loader(pretrained=True, pretrained_model_path=args.weights)
    model = model.cuda()
    model.eval()

    names = load_imagenet_class_names()

    pil = Image.open(img_path).convert("RGB")
    x = build_preprocess()(pil).unsqueeze(0).cuda()

    with torch.no_grad():
        out = model.forward(x, y=None)

    logits = out["logits_tr"]
    nll = out.get("nll_joint_tr")
    if nll is not None and nll.dim() > 0:
        nll_mean = nll.mean().item()
    else:
        nll_mean = float(nll) if nll is not None else float("nan")

    probs = logits.softmax(dim=1)
    k = min(args.topk, logits.shape[1])
    top = probs.topk(k, dim=1)

    print()
    print("Image:", str(img_path))
    print(f"Batch nll_joint (mean over batch): {nll_mean:.6f}")
    print()
    print(f"Top-{k} predictions:")
    for i in range(k):
        idx = int(top.indices[0, i].item())
        pval = float(top.values[0, i].item())
        label = names[idx] if idx < len(names) else f"class_{idx}"
        print(f"  {i + 1}. {label}  (index={idx}, prob={pval:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
