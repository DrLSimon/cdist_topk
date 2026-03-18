from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from intrinsic_dim.synthetic.sampling import sample_patches, list_manifolds, list_densities, get_max_dim


def test_sampling_api():
    assert 'linear' in list_manifolds()
    assert 'gaussian' in list_densities()
    dims = torch.tensor([[1, 2]])
    samples = sample_patches(dims, patch_size=1, nb_channels=8, n_samples=10, manifold='linear')
    assert samples.shape[:3] == (1, 2, 10)
    assert get_max_dim('linear', 8) == 8


def main():
    test_sampling_api()
    print("✓ test_sampling_api passed")


if __name__ == '__main__':
    main()
