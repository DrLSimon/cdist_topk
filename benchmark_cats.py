import torch
from cdist_topk import chunked_cdist_topk, reference_cdist_topk, iterative_chunked_cdist_topk
from utils import gpu_memory_tracker, timer, load_cats, make_loader, images_to_patches

@timer
@gpu_memory_tracker
def cdist_topk_chunked(x, y, chunk_size=100):
    vals, _ = chunked_cdist_topk(x, y, k=10, chunk_size=chunk_size, approx=False)
    return vals

@timer
@gpu_memory_tracker
def cdist_topk_iterative(x, y, chunk_size=100):
    vals, _ = iterative_chunked_cdist_topk(x, y, k=10, chunk_size=chunk_size, approx=False)
    return vals

@timer
@gpu_memory_tracker
def cdist_topk(x, y):
    vals, _ = reference_cdist_topk(x, y, k=10, approx=False)
    return vals



def compare_variants(all_cats, Nx, Ny):
    
    some_cats_x = all_cats[:Nx]
    some_cats_y = all_cats[Nx:(Nx+Ny)]
    cats_patches_x = images_to_patches(some_cats_x).cuda().float()/255
    cats_patches_y = images_to_patches(some_cats_y).cuda().float()/255
    iter_vals = cdist_topk_iterative(cats_patches_x, cats_patches_y, debug=True)
    chunk_vals = cdist_topk_chunked(cats_patches_x, cats_patches_y, debug=True)
    ref_vals = cdist_topk(cats_patches_x, cats_patches_y, debug=True)
    assert torch.allclose(ref_vals, iter_vals, atol=1e-6)
    assert torch.allclose(ref_vals, chunk_vals, atol=1e-6)
    print("✓ correctness tests passed")


def main():
    all_cats = load_cats()
    compare_variants(all_cats, Nx=100, Ny=1000)

if __name__=='__main__':
    main()


    

