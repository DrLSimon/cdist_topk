import torch
from cdist_topk import chunked_cdist_topk, reference_cdist_topk, iterative_chunked_cdist_topk
from utils import gpu_memory_tracker, timer, load_cats

def images_to_patches(x: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """
    Convert batch of RGB images to batch of patches.
    
    Args:
        x: Input tensor of shape (N, 3, H, W)
        patch_size: Size of each patch (default: 32)
    
    Returns:
        Tensor of shape (N, 3*patch_size*patch_size, H//patch_size, W//patch_size)
        e.g. (N, 3, 512, 512) -> (N, 3072, 16, 16)
    """
    N, C, H, W = x.shape
    p = patch_size
    Ph, Pw = H // p, W // p  # number of patches: 32x32 = 1024

    return (
        x.reshape(N, C, Ph, p, Pw, p)  # (N, C, Ph, p, Pw, p)
         .permute(1, 3, 5, 0, 2, 4)    # (3, p, p, N,  Ph, Pw)
         .reshape(C, p, p, N, Ph*Pw)   # (3, p, p, N,  Ph*Pw)
    )

@timer
@gpu_memory_tracker
def cdist_topk_chunked(x):
    chunked_cdist_topk(x, x, k=10, chunk_size=100)

@timer
@gpu_memory_tracker
def cdist_topk_iterative(x):
    iterative_chunked_cdist_topk(x, x, k=10, chunk_size=100)

@timer
@gpu_memory_tracker
def cdist_topk(x):
    reference_cdist_topk(x, x, k=10)



def main():
    all_cats = load_cats().bfloat16()
    N = 600
    some_cats = all_cats[:N]
    cats_patches = images_to_patches(some_cats).cuda()
    cdist_topk_iterative(cats_patches)
    cdist_topk_chunked(cats_patches)
    cdist_topk(cats_patches)

if __name__=='__main__':
    main()


    

