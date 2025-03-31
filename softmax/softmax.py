import triton
import triton.language as tl
import triton
import triton.language as tl
import torch
@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xb, stride_xm, stride_xn,
    stride_ob, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_xn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % BLOCK_SIZE_N
    offset =  pid_b * stride_xb + offset_xm[:, None] * stride_xm + offset_xn[None, :] * stride_xn
    x_ptr += offset
    mask = (offset_xm[:, None] < M) & (offset_xn[None, :] < N)
    # notation is from flash attrntion 2 paper
    s = tl.load(x_ptr, mask=mask, other = -float('inf'))
    m = tl.max(s, axis=1, keep_dims=True)
    p = tl.exp(s - m)
    l = tl.sum(p, axis=1, keep_dims=True)  # correction factor = 0

    output_offset = ( pid_b * stride_ob + 
        offset_xm[:, None] * stride_om
        + (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_on
    )
    tl.store(output_ptr + output_offset, p / l, mask=mask)

def softmax(x: torch.Tensor):
    B, M, N = x.shape
    output = torch.zeros((B, M, N), device = 'cuda', dtype = torch.float32)
    assert x.is_cuda
    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M']))
    softmax_kernel[grid](x, output,
                         M, N,      
                    *x.stride(), 
                    *output.stride(), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = triton.next_power_of_2(N))   
    return output