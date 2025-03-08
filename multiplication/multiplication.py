import triton
import triton.language as tl
import torch
print("is cuda being used?", torch.cuda.is_available())


@triton.jit
def dot_sum_kernel(x_ptr, output_ptr, n, SUM_BLOCK_SIZE: tl.constexpr):
    # these values are the same for both x and y
    pid = tl.program_id(axis = 0) # row vector pids
    block_start = pid * SUM_BLOCK_SIZE
    stride = tl.arange(0, SUM_BLOCK_SIZE) 
    offset = block_start + stride

    mask = offset < n

    x = tl.load(x_ptr + offset, mask = mask)
    output = tl.sum(x, axis = 0)
    tl.atomic_add(output_ptr, output) # no mask!

def dot_sum(x: torch.Tensor):
    output = torch.zeros((1, ), device = 'cuda')
    assert x.is_cuda and output.is_cuda
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['SUM_BLOCK_SIZE']), )
    dot_sum_kernel[grid](x, output, n, SUM_BLOCK_SIZE = 16)   
    return output 

n = 300*32
x = torch.rand((1,n), device = 'cuda')
# output_torch = torch.sum(x)
output_triton = dot_sum(x)
output_triton.shape