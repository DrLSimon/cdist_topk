from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from intrinsic_dim.data.patches import images_to_patches, patches_to_images


def test_patch_roundtrip():
    x = torch.randn(5, 3, 32, 32)
    patches = images_to_patches(x, patch_size=8)
    x2 = patches_to_images(patches, patch_size=8)
    assert torch.allclose(x, x2)


def main():
    test_patch_roundtrip()
    print("✓ test_patch_roundtrip passed")


if __name__ == '__main__':
    main()
