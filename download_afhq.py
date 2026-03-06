import torch
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import io

# ── Configuration ──────────────────────────────────────────────────────────────
RESIZE_TO = (512, 512)   # resize all images to this (H, W); set None to keep originals
OUTPUT_FILE = "cats_tensor.pt"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading huggan/AFHQ dataset from Hugging Face…")
    # The dataset has a 'label' column: 0=cat, 1=dog, 2=wild
    # We load both splits and filter for cats.
    dataset = load_dataset("huggan/AFHQ")

    transform = transforms.Compose([
        transforms.Resize(RESIZE_TO) if RESIZE_TO else transforms.Lambda(lambda x: x),
        transforms.PILToTensor(),          # → (3, H, W), uint8 
    ])

    tensors = []
    total = 0

    for split_name, split_data in dataset.items():
        print(f"\nProcessing split '{split_name}' ({len(split_data)} samples)…")

        # Filter cats — label == 0
        # (AFHQ label mapping: cat=0, dog=1, wild=2)
        cats = split_data.filter(lambda ex: ex["label"] == 0)
        print(f"  Found {len(cats)} cat images.")

        for i, example in enumerate(cats):
            img = example["image"]                      # PIL Image
            if not isinstance(img, Image.Image):
                # Some dataset versions store raw bytes
                img = Image.open(io.BytesIO(img)).convert("RGB")
            else:
                img = img.convert("RGB")

            tensors.append(transform(img))

            if (i + 1) % 100 == 0:
                print(f"    Processed {i + 1}/{len(cats)}…", end="\r")

        total += len(cats)

    if not tensors:
        print("\nNo cat images found — check that the dataset labels are correct.")
        return

    cats_tensor = torch.stack(tensors)   # (N, 3, H, W)
    print(f"\n\nFinal tensor shape : {cats_tensor.shape}")
    print(f"dtype              : {cats_tensor.dtype}")
    print(f"Value range        : [{cats_tensor.min():.3f}, {cats_tensor.max():.3f}]")

    torch.save(cats_tensor, OUTPUT_FILE)
    print(f"\nSaved to '{OUTPUT_FILE}' ✓")


if __name__ == "__main__":
    main()
