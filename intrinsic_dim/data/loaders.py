import torch
import tqdm
from torch.utils.data import DataLoader

def make_loader(x: torch.Tensor, batch_dim: int = -2, **kwargs) -> DataLoader:
    use_tqdm = kwargs.pop('tqdm', False)
    transform = kwargs.pop('transform', None)
    device = kwargs.pop('device', None)
    batch_dim = batch_dim % x.ndim
    x_perm = x.moveaxis(batch_dim, 0)
    def collate(samples):
        stacked = torch.stack(samples, dim=0)
        out = stacked.moveaxis(0, batch_dim)
        if device: out = out.to(device)
        if transform: out = transform(out)
        return out
    loader = DataLoader(x_perm, collate_fn=collate, **kwargs)
    return tqdm.tqdm(loader, leave=False) if use_tqdm else loader
