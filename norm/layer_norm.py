import triton
import triton.language as tl
import torch
torch.cuda.is_available()
torch.set_printoptions(profile="full")
@triton.jit
def layer_norm_kernel(x_ptr,output_ptr, eps,
                M, N: tl.constexpr,
                stride_xm, stride_xn,
                stride_om, stride_on,
                BLOCK_SIZE_M: tl.constexpr):

    pid_m = tl.program_id(axis = 0) 
    pid_n = tl.program_id(axis = 1) 
    offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_xm +  (pid_n * N + tl.arange(0, N))[None,:]*stride_xn
    x = tl.load(x_ptr + offset)   
       
    mean = tl.sum(x, axis = -1, keep_dims = True) / M
    variance = tl.sum((x- mean)*(x- mean), axis = -1, keep_dims = True)/N
    output_offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] *stride_om + (pid_n * N + tl.arange(0, N))[None, :] *stride_on
    output = tl.load(output_ptr + offset)   
    output = (x - mean) / tl.sqrt(variance + eps)        
    tl.store(output_ptr + output_offset,  output)

def layer_norm(x: torch.Tensor):
    M , N = x.shape
    output = torch.zeros((M, N),  device = 'cuda', dtype = torch.float32 )
    assert x.is_cuda and output.is_cuda
    eps = 1e-6
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), 1)
    layer_norm_kernel[grid](x, output, eps,
                    M, N, 
                    x.stride(0), x.stride(1),
                    output.stride(0), output.stride(1),
                    BLOCK_SIZE_M = 32, 
                    )
    return output