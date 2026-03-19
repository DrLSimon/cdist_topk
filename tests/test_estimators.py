from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from intrinsic_dim.synthetic.sampling import sample_patches
from intrinsic_dim import get_estimator, list_estimators

def test_list_estimators():
    estimators = list_estimators()
    assert isinstance(estimators, list)
    assert all(isinstance(name, str) for name in estimators)
    assert "pca" in estimators
    assert "mle" in estimators
    assert "mle_avg" in estimators
    print("✓ test_list_estimators passed")

def test_estimators_shapes():
    pca_estimator = get_estimator("pca", threshold=0.95)
    mle_estimator = get_estimator("mle", k=5, n_anchors=50)
    mle_avg_estimator = get_estimator("mle_avg", k=5, n_anchors=50)
    dims = torch.tensor([[1, 2, 4]])
    samples = sample_patches(dims, patch_size=1, nb_channels=8, n_samples=200, manifold='linear')
    pca = pca_estimator(samples)
    mle = mle_estimator(samples)
    mle_avg = mle_avg_estimator(samples)
    assert tuple(pca.shape) == (1, 3)
    assert tuple(mle.shape) == (1, 3)
    assert tuple(mle_avg.shape) == (1, 3)
    print("✓ test_estimators_shapes passed")


def main():
    test_list_estimators()
    test_estimators_shapes()


if __name__ == '__main__':
    main()
