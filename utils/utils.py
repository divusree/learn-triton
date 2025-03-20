import triton
import triton.language as tl
import torch
torch.set_printoptions(profile="full")
"""
add script to path as  
import sys
sys.path.insert(0,'..')
import the functions as normal
"""
@triton.jit
def power_kernel(x_ptr,output_ptr, y,
                    M: tl.constexpr , N: tl.constexpr , 
                    stride_xm, stride_xn, 
                    BLOCK_SIZE_M: tl.constexpr, 
                    BLOCK_SIZE_N: tl.constexpr 
                    ):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset =  offset_xm[:,None] * stride_xm + offset_xn[None,:] *stride_xn
    x_ptr += offset
    mask = (offset_xm[:,None] < M) & (offset_xn[None, :] < N)
    x = tl.load(x_ptr, mask = mask)
    output = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), value = 1, dtype = tl.float32)
    for i in range(y):
        output = output * x
    tl.store(output_ptr + offset, output, mask = mask )

def power(x: torch.Tensor, y):
    M, N = x.shape
    output = torch.zeros((M, N), device = 'cuda', dtype = torch.float32)
    assert x.is_cuda
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    power_kernel[grid](x, output, y,
                         M, N,      
                    x.stride(0), x.stride(1), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = 32)   
  
    return output