import os
import subprocess
from pathlib import Path
import torch

from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import io

# ── Configuration ──────────────────────────────────────────────────────────────
RESIZE_TO = (512, 512)   # resize all images to this (H, W); set None to keep originals
# ──────────────────────────────────────────────────────────────────────────────

def load_cats(file_path: str | os.PathLike | None = None):
    project_root = Path(__file__).resolve().parents[2]
    file_path = Path(file_path) if file_path is not None else project_root / "cats_tensor.pt"
    if not file_path.exists():
        dataset = download_afhq()
        save_category("cat", dataset, file_path)
    return torch.load(file_path)


def download_afhq():
    print("Loading huggan/AFHQ dataset from Hugging Face…")
    dataset = load_dataset("huggan/AFHQ")
    return dataset


def save_category(label : str | int, dataset, file_path: str | os.PathLike):
    if isinstance(label, str):
        categories={"cat" : 0, "dog" : 1, "wild" : 2}
        label = categories.get(label.lower())
    transform = transforms.Compose([
        transforms.Resize(RESIZE_TO) if RESIZE_TO else transforms.Lambda(lambda x: x),
        transforms.PILToTensor(),          # → (3, H, W), uint8 
    ])
    tensors = []
    total = 0

    for split_name, split_data in dataset.items():
        print(f"\nProcessing split '{split_name}' ({len(split_data)} samples)…")

        # Filter category
        samples = split_data.filter(lambda ex: ex["label"] == 0)
        print(f"  Found {len(samples)} {label} images.")

        for i, example in enumerate(samples):
            img = example["image"]                      # PIL Image
            if not isinstance(img, Image.Image):
                # Some dataset versions store raw bytes
                img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            tensors.append(transform(img))

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(samples)}…", end="\r")

        total += len(samples)

    if not tensors:
        print(f"\nNo {label} images found — check that the dataset labels are correct.")
        return

    samples_as_tensor = torch.stack(tensors)   # (N, 3, H, W)
    print(f"\n\nFinal tensor shape : {samples_as_tensor.shape}")
    print(f"dtype              : {samples_as_tensor.dtype}")
    print(f"Value range        : [{samples_as_tensor.min():.3f}, {samples_as_tensor.max():.3f}]")

    torch.save(samples_as_tensor, file_path)
    print(f"\nSaved to '{file_path}' ✓")
