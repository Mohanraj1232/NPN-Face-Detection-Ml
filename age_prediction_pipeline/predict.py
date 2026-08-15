"""
Step 5 — run both predictions on a single image and print them side by side.

Usage:
    python predict.py --checkpoint runs/exp1/best_model.pt --image path\\to\\face.jpg
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from data import build_transforms
from model import DualHeadAgeModel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    min_age = ckpt["min_age"]

    model = DualHeadAgeModel(num_classes=ckpt["num_classes"], backbone=ckpt["backbone"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = build_transforms(train=False, img_size=ckpt["img_size"])
    img = Image.open(args.image).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, reg_out = model(x)
        probs = F.softmax(class_logits, dim=1)[0]

    top5 = torch.topk(probs, k=5)
    top5_ages = [(int(idx) + min_age, float(p)) for idx, p in zip(top5.indices, top5.values)]
    class_pred_age = top5_ages[0][0]
    reg_pred_age = float(reg_out.item())

    print(f"\nImage: {args.image}")
    print(f"Classification prediction: {class_pred_age} years (confidence {top5_ages[0][1]:.1%})")
    print("  Top-5 candidates: " + ", ".join(f"{a} ({p:.1%})" for a, p in top5_ages))
    print(f"Regression prediction:     {reg_pred_age:.1f} years")
    print(f"Agreement (|difference|):  {abs(class_pred_age - reg_pred_age):.1f} years")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/exp1/best_model.pt"))
    ap.add_argument("--image", type=Path, required=True)
    args = ap.parse_args()
    main(args)
