import triton
import triton.language as tl
import torch
@triton.jit

def rmax_kernel(x_ptr, output_ptr,
                    M, N, 
                    stride_xm, stride_xn, 
                    stride_om, stride_on, 
                    BLOCK_SIZE_M: tl.constexpr, 
                    BLOCK_SIZE_N: tl.constexpr 
                    ):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    offset =  offset_xm[:,None] * stride_xm +  offset_xn[None,:]*stride_xn
    x = tl.load(x_ptr + offset, mask = (offset_xm[:,None] < M) & (offset_xn[None,:] < N), other = 0)
    m = tl.max(x, axis = 1) # axis = 1 gives me row wise values
    tl.atomic_max(output_ptr + offset_xm * stride_om, m)

def rowmax(x: torch.Tensor):
    M, N = x.shape
    output = torch.zeros((x.shape[0], 1), device = 'cuda', dtype = torch.float32)
    assert x.is_cuda and output.is_cuda
    n = x.numel()
    grid = lambda meta: (triton.cdiv(output.shape[0], meta['BLOCK_SIZE_M']), triton.cdiv(output.shape[1], meta['BLOCK_SIZE_N']))
    rmax_kernel[grid](x, output, M, N, 
                    x.stride(0), x.stride(1), 
                    output.stride(0), output.stride(1), 
                    BLOCK_SIZE_M = 1,
                    BLOCK_SIZE_N = 32)   