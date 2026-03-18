import os
import subprocess
from pathlib import Path
import torch

def load_cats(file_path: str | os.PathLike | None = None):
    project_root = Path(__file__).resolve().parents[2]
    file_path = Path(file_path) if file_path is not None else project_root / 'cats_tensor.pt'
    if not file_path.exists():
        script = project_root / 'scripts' / 'download_afhq.py'
        subprocess.run(["python", str(script)], check=True, cwd=str(project_root))
    return torch.load(file_path)
