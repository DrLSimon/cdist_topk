import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intrinsic_dim.data.afhq import load_cats


@pytest.fixture(scope='session')
def all_cats():
    try:
        return load_cats()
    except Exception as e:
        pytest.skip(f'AFHQ cats dataset unavailable: {e}')
