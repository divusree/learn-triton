import triton
import triton.language as tl
import torch
@triton.jit
def softmax_kernel(x_ptr, m_ptr, l_ptr,output_ptr,
                    M, N, 
                    stride_xm, stride_xn, 
                    stride_mm, stride_mn, 
                    stride_lm, stride_ln, 
                    stride_om, stride_on, 
                    BLOCK_SIZE_M: tl.constexpr, 
                    BLOCK_SIZE_N: tl.constexpr 
                    ):
    pid_m = tl.program_id(axis = 0)
    pid_n = tl.program_id(axis = 1)
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset =  offset_xm[:,None] * stride_xm + offset_xn[None,:] *stride_xn
    x_ptr += offset
    
    #  m is row wise max of x (per block)
    m = tl.full((BLOCK_SIZE_M, 1), value = -torch.inf, dtype = tl.float32) # axis = 1 gives me row wise values
    l = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32)
    # output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for i in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        x = tl.load(x_ptr) #, mask = (offset < N), other = 0)

        prev = m
        m = tl.max(tl.maximum(m, x), axis = 1, keep_dims = True) # axis = 1 gives me row wise values
        p = tl.exp(x - m)
        corr = tl.exp(prev - m)
        l = l * corr + tl.sum(p, axis = 1, keep_dims = True)

        x_ptr += BLOCK_SIZE_N *  stride_xn

    tl.store(m_ptr + offset_xm[:,None] * stride_mm, m)
    tl.store(l_ptr + offset_xm[:,None] * stride_lm, l)

def softmax(x: torch.Tensor):
    M, N = x.shape
    m = torch.zeros((M, 1), device = 'cuda', dtype = torch.float32)
    l = torch.zeros((M, 1), device = 'cuda', dtype = torch.float32)
    output = torch.zeros((M, N), device = 'cuda', dtype = torch.float32)
    assert x.is_cuda and m.is_cuda and l.is_cuda
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    softmax_kernel[grid](x, m, l, output,
                         M, N,      
                    x.stride(0), x.stride(1), 
                    m.stride(0), m.stride(1), 
                    l.stride(0), l.stride(1), 
                    output.stride(0), output.stride(1), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = 32)   
  
    return m, l, output