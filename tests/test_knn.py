from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from intrinsic_dim.neighbors.knn import raw_topk_dists, compute_dataset_topk_dists
from intrinsic_dim.data.loaders import make_loader


def test_dataset_topk_matches_raw():
    x = torch.randn(2, 3, 20, 4)
    ref = raw_topk_dists(x, x, k=5)
    loaderx = make_loader(x, batch_dim=-2, batch_size=7)
    loadery = make_loader(x, batch_dim=-2, batch_size=9)
    got = compute_dataset_topk_dists(loaderx, loadery, k=5)
    assert torch.allclose(ref, got)


def main():
    test_dataset_topk_matches_raw()
    print("✓ test_dataset_topk_matches_raw passed")


if __name__ == '__main__':
    main()
