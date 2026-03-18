from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from intrinsic_dim.synthetic.sampling import sample_patches
from intrinsic_dim.estimators.pca import compute_pca_dims
from intrinsic_dim.estimators.mle import compute_mle_dims


def test_estimators_shapes():
    dims = torch.tensor([[1, 2, 4]])
    samples = sample_patches(dims, patch_size=1, nb_channels=8, n_samples=200, manifold='linear')
    pca = compute_pca_dims(samples, threshold=0.95)
    mle, mle_avg = compute_mle_dims(samples, k=5, n_anchors=50)
    assert tuple(pca.shape) == (1, 3)
    assert tuple(mle.shape) == (1, 3)
    assert tuple(mle_avg.shape) == (1, 3)


def main():
    test_estimators_shapes()
    print("✓ test_estimators_shapes passed")


if __name__ == '__main__':
    main()
