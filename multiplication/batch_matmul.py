import triton
import triton.language as tl
import torch
torch.cuda.is_available()
import os
import json
os.environ['TRITON_PRINT_AUTOTUNING'] = "1"
def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'])
@triton.jit
def batch_matmul_kernel(
                # my pointers
                x_ptr,y_ptr,output_ptr,
                # matrix dims
                B, M, K, N,
                # strides for each matrix
                stride_xb, stride_xm, stride_xk,
                stride_yb, stride_yk, stride_yn,
                stride_ob, stride_om, stride_on,
                # BLOCK_SIZE for all dims - for simplicity
                BLOCK_SIZE_B: tl.constexpr, 
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, 
                BLOCK_SIZE_K: tl.constexpr, 
                # GROUP_SIZE for all dims - for simplicity
                GROUP_SIZE_M: tl.constexpr):

    # with batch matmul we have to always start at axis = 1 for rows. 
    # pid = tl.program_id(axis=1)
    # num_pid_m = tl.cdiv(B, BLOCK_SIZE_B)
    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # pid_n = (pid % num_pid_in_group) // group_size_m    

    # comment the above and uncomment below for non grouped matmul
    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1) 
    pid_n = tl.program_id(axis=2) 

    # strides for the M, K and N dim. 
    # A[b, i, j] = b * stride_Ab  + i *stride_Am + j * stride_An
    # &A[b, m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + A.stride(0) + (m : m+BLOCK_SIZE_M)*A.stride(1) + (k : k+BLOCK_SIZE_K)*A.stride(2);
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    x_ptr += pid_b * stride_xb + offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += pid_b * stride_yb +  offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptr, mask = offset_k [None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        y = tl.load(y_ptr, mask = offset_k[:,None] < K - k * BLOCK_SIZE_K, other = 0.0)            
        accumulator = tl.dot(x, y, accumulator)
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K *  stride_xk
        y_ptr += BLOCK_SIZE_K *  stride_yk
        
    accumulator = accumulator.to(tl.float16)
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offset = pid_b * stride_ob +  offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    output_mask =  (offset_om[:,None] < M) & ( offset_on[None, :] < N)
    tl.store(output_ptr + output_offset, accumulator, mask = output_mask)
    
def matmul(x: torch.Tensor, y: torch.Tensor, dtype = None):
    B, M , K = x.shape
    B1, K1, N = y.shape
    assert (K == K1) & (B == B1)
    output = torch.empty((B, M, N), device = 'cuda', dtype = dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    grid = lambda meta: (triton.cdiv(B, meta['BLOCK_SIZE_B']), triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    batch_matmul_kernel[grid](x, y, output, 
                        B, M, K, N, 
                        x.stride(0), x.stride(1), x.stride(2), 
                        y.stride(0), y.stride(1), y.stride(2), 
                        output.stride(0), output.stride(1), output.stride(2),

                        BLOCK_SIZE_B = 1, # 1 batch per block 
                        # uncomment if not autotuning
                        # BLOCK_SIZE_M = 32, 
                        # BLOCK_SIZE_N = 32, 
                        # BLOCK_SIZE_K = 32, 
                        # GROUP_SIZE along M dim
                        GROUP_SIZE_M = 8
                        )
    return output

# dtype = torch.float32
# x = torch.rand((64, 1024, 64), device = 'cuda', dtype = dtype)
# y = torch.rand((64, 64, 384), device = 'cuda', dtype = dtype)
# output_triton = matmul(x, y, dtype = dtype).to(torch.float16)
# output_torch = torch.matmul(x, y).to(torch.float16)
# print(f"Are the results close? {torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0)}")
