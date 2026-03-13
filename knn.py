import torch


def _topk_vals(d, k):
    _k = min(k, d.shape[-1])
    vals, _ = torch.topk(d, _k, dim=-1, largest=False, sorted=True)
    return vals


def raw_topk_dists(x, y, k):
    d = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    return _topk_vals(d, k)


def update_topk_dists_from_batch(x, y, k, topk_dists):
    d = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    new_dists = _topk_vals(d, k)
    if topk_dists is None:
        return new_dists
    return _topk_vals(torch.cat([topk_dists, new_dists], dim=-1), k)


def concat_topk_dists(topk_dists, batch_topk_dists):
    if topk_dists is None:
        return batch_topk_dists
    return torch.cat([topk_dists, batch_topk_dists], dim=-2)


def compute_dataset_topk_dists(loaderx, loadery, k=25):
    """
    Compute top-k nearest neighbour distances across all patches in a dataset.

    Args:
        loaderx: Iterable of x-batches, each of shape (..., Bx, D).
        loadery: Iterable of y-batches, each of shape (..., By, D).
        k:       Number of nearest neighbours.

    Returns:
        Tensor of shape (N, k) containing sorted top-k distances.
    """
    all_topk_dists = None
    for x in loaderx:
        batch_topk_dists = None
        for y in loadery:
            batch_topk_dists = update_topk_dists_from_batch(x, y, k, batch_topk_dists)
        all_topk_dists = concat_topk_dists(all_topk_dists, batch_topk_dists)
    return all_topk_dists