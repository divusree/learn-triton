import triton
import triton.language as tl
import torch
print("is cuda being used?", torch.cuda.is_available())

@triton.jit
def matmul_kernel_naive(x_ptr,y_ptr,output_ptr,
                M, K, N,
                stride_xm, stride_xk,
                stride_yk, stride_yn,
                stride_om, stride_on,
                BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(axis = 0) * BLOCK_SIZE
    pid_n = tl.program_id(axis = 1) * BLOCK_SIZE
    accumulator = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = tl.float32)
    for k in range(0, K, BLOCK_SIZE):
        # &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)*A.stride(0) + (k : k+BLOCK_SIZE_K)*A.stride(1);
        # &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)*B.stride(0) + (n : n+BLOCK_SIZE_N)*B.stride(1);
        offset_xm = pid_m + tl.arange(0, BLOCK_SIZE)
        offset_k =  tl.arange(0, BLOCK_SIZE)
        offset_yn = pid_n + tl.arange(0, BLOCK_SIZE)
        x_block = offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
        y_block = offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

        # load and compute
        x = tl.load(x_ptr + x_block, mask = (offset_xm [:,None]< M) & (offset_k[None,:] < K))
        y = tl.load(y_ptr + y_block, mask = (offset_yn[:,None] < N) & (offset_k[None,:] < K))
        accumulator = tl.dot(x, y, accumulator)
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE * stride_xk
        y_ptr += BLOCK_SIZE * stride_yk

    offset_om = pid_m + tl.arange(0, BLOCK_SIZE)
    offset_on = pid_n + tl.arange(0, BLOCK_SIZE)
    output_offset = offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    tl.store(output_ptr + output_offset, accumulator, mask = (offset_om[:,None] < M) &( offset_on[None, :] < N))

def matmul(x: torch.Tensor, y: torch.Tensor):
    M , K = x.shape
    K1, N = y.shape
    assert K == K1
    output = torch.empty((M, N), device = 'cuda')
    assert x.is_cuda and y.is_cuda and output.is_cuda

    grid = lambda meta: ((triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE'])))
    matmul_kernel_naive[grid](x, y, output, 
                        M, K, N, 
                        x.stride(0), x.stride(1),
                        y.stride(0), y.stride(1),
                        output.stride(0), output.stride(1),
                        BLOCK_SIZE = 16)
    return output