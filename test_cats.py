import torch
import tqdm
from cdist_topk import chunked_cdist_topk, reference_cdist_topk, iterative_chunked_cdist_topk
from utils import gpu_memory_tracker, timer, load_cats, make_loader, images_to_patches

def raw_topk_dists(x, y, k):
    d = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    vals, _ = torch.topk(d, k, dim=-1, largest=False, sorted=True)
    return vals


def update_topk_dists_from_batch(x, y, k, topk_dists):
    def extract_topk(d):
        _k = min(k, d.shape[-1])
        vals, _ = torch.topk(d, _k, dim=-1, largest=False, sorted=True)
        return vals

    d = torch.cdist(x, y, compute_mode='donot_use_mm_for_euclid_dist')
    new_dists = extract_topk(d)
    if topk_dists is None:
        return new_dists
    all_dists=torch.cat([topk_dists, new_dists], dim=-1)
    topk_dists = extract_topk(all_dists)
    return topk_dists



def concat_topk_dists(topk_dists, batch_topk_dists):
    if topk_dists is None:
        return batch_topk_dists
    return torch.cat([topk_dists, batch_topk_dists], dim=-2)

def run_on_full_dataset(all_cats, k=25, patch_size=32, bsx=256, bsy=128):
    all_cats_patches = images_to_patches(all_cats, patch_size=patch_size)
    loaderx = make_loader(all_cats_patches, batch_dim=-2, batch_size=bsx)
    loadery = make_loader(all_cats_patches, batch_dim=-2, batch_size=bsy)
    
    all_topk_dists = None

    for x in tqdm.tqdm(loaderx):
        x = x.cuda().float()/255
        batch_topk_dists = None
        for y in loadery:
            y = y.cuda().float()/255
            batch_topk_dists = update_topk_dists_from_batch(x, y, k, batch_topk_dists)
        all_topk_dists = concat_topk_dists(all_topk_dists, batch_topk_dists)
    return all_topk_dists


def test_correctness(all_cats, patch_size=32):
    some_cats = all_cats[:100]
    some_cats_patches = images_to_patches(some_cats, patch_size).cuda().float()/255
    ref_dists = raw_topk_dists(some_cats_patches, some_cats_patches, k=25)
    topk_dists =  run_on_full_dataset(some_cats, k=25, patch_size=patch_size, bsx=7, bsy=9)
    assert torch.allclose(ref_dists, topk_dists, atol=1e-6)
    print("✓ correctness tests passed")

def main():
    all_cats = load_cats()
    test_correctness(all_cats)
    all_topk_dists = run_on_full_dataset(all_cats)
    torch.save(all_topk_dists, 'top25_dists.pt')

if __name__=='__main__':
    main()


    

