import triton
import triton.language as tl
import torch

@triton.jit
def sum_kernel(x_ptr, output_ptr, n, SUM_BLOCK_SIZE: tl.constexpr):
    # these values are the same for both x and y
    pid = tl.program_id(axis = 0) # row vector pids
    block_start = pid * SUM_BLOCK_SIZE
    stride = tl.arange(0, SUM_BLOCK_SIZE) 
    offset = block_start + stride

    mask = offset < n

    x = tl.load(x_ptr + offset, mask = mask)
    output = tl.sum(x, axis = 0)
    tl.atomic_add(output_ptr, output) # no mask!

def sum1D(x: torch.Tensor):
    output = torch.zeros((1, ), device = 'cuda')
    assert x.is_cuda and output.is_cuda
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['SUM_BLOCK_SIZE']), )
    sum_kernel[grid](x, output, n, SUM_BLOCK_SIZE = 16)   
    return output 