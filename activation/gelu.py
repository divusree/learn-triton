import triton
import triton.language as tl
import torch
import math
torch.set_printoptions(profile="full")
@triton.jit
def gelu_kernel(x_ptr,output_ptr, invsqrt2,
                M, N: tl.constexpr,
                stride_xm, stride_xn,
                stride_om, stride_on,
                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr
                ):

    pid_m = tl.program_id(axis = 0) 
    pid_n = tl.program_id(axis = 1) 
    offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_xm +  (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None,:]*stride_xn
    x = tl.load(x_ptr + offset)   
       
    output = x * 0.5 * (1 + tl.erf(x * invsqrt2 ))

    tl.store(output_ptr + offset,  output)

def gelu(x: torch.Tensor):
    M , N = x.shape
    output = torch.zeros((M, N),  device = 'cuda', dtype = torch.float32 )
    assert x.is_cuda and output.is_cuda
    invsqrt2 = math.sqrt(2)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    gelu_kernel[grid](x, output, invsqrt2,
                    M, N, 
                    x.stride(0), x.stride(1),
                    output.stride(0), output.stride(1),
                    BLOCK_SIZE_M = 32, 
                    BLOCK_SIZE_N = 32, 
                    )
    return output