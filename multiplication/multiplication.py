import triton
import triton.language as tl
import torch
torch.cuda.is_available()
import os
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
    offset_xm = pid_m + tl.arange(0, BLOCK_SIZE)
    offset_k =  tl.arange(0, BLOCK_SIZE)
    offset_yn = pid_n + tl.arange(0, BLOCK_SIZE)
    x_ptr += offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    for k in range(0, K, BLOCK_SIZE):
        # &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)*A.stride(0) + (k : k+BLOCK_SIZE_K)*A.stride(1);
        # &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)*B.stride(0) + (n : n+BLOCK_SIZE_N)*B.stride(1);

        # load and compute
        x = tl.load(x_ptr, mask = (offset_k[None,:] < K-k*BLOCK_SIZE), other = 0.0)
        y = tl.load(y_ptr, mask =(offset_k[:, None] < K-k*BLOCK_SIZE), other = 0.0)
        accumulator = tl.dot(x, y, accumulator)
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE * stride_xk
        y_ptr += BLOCK_SIZE * stride_yk

    accumulator = accumulator.to(tl.float16)
    offset_om = pid_m + tl.arange(0, BLOCK_SIZE)
    offset_on = pid_n + tl.arange(0, BLOCK_SIZE)
    output_offset = offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    tl.store(output_ptr + output_offset, accumulator, mask = (offset_om[:,None] < M) &( offset_on[None, :] < N))


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'])
@triton.jit
def group_matmul_kernel(
                # my pointers
                x_ptr,y_ptr,output_ptr,
                # matrix dims
                M, K, N,
                # strides for each matrix
                stride_xm, stride_xk,
                stride_yk, stride_yn,
                stride_om, stride_on,
                # BLOCK_SIZE for all dims - for simplicity
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, 
                BLOCK_SIZE_K: tl.constexpr, 
                # GROUP_SIZE for all dims - for simplicity
                GROUP_SIZE_M: tl.constexpr):

    # recall previously we had row major ordering and 
    # pid_m = tl.program_id(axis = 0) 
    # pid_n = tl.program_id(axis = 1) 

    # we get new pid_m and pid_n after we group pids together
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m    

    pid_m *= BLOCK_SIZE_M 
    pid_n *= BLOCK_SIZE_N 

    # the rest after getting pid_m and pid_n are as before
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    offset_xm = (pid_m + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n + tl.arange(0, BLOCK_SIZE_N)) % N
    x_ptr += offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)*A.stride(0) + (k : k+BLOCK_SIZE_K)*A.stride(1);
        # &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)*B.stride(0) + (n : n+BLOCK_SIZE_N)*B.stride(1);

        # load and compute. mask: K - k * BLOCK_SIZE_K = number of values to load for that block. the rest get masked
        x = tl.load(x_ptr, mask = offset_k [None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        y = tl.load(y_ptr, mask = offset_k[:,None] < K - k * BLOCK_SIZE_K, other = 0.0)    
        accumulator = tl.dot(x, y, accumulator)
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K * stride_xk
        y_ptr += BLOCK_SIZE_K * stride_yk

    accumulator = accumulator.to(tl.float16)
    offset_om = pid_m + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n + tl.arange(0, BLOCK_SIZE_N)
    output_offset = offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    tl.store(output_ptr + output_offset, accumulator, mask = (offset_om[:,None] < M) &( offset_on[None, :] < N))
def matmul3D(x: torch.Tensor, y: torch.Tensor, dtype = None):
    M , K = x.shape
    K1, N = y.shape
    assert K == K1
    output = torch.empty((M, N), device = 'cuda', dtype = dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda


    # grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE'])*triton.cdiv(N, meta['BLOCK_SIZE']),)
    # matmul_kernel_naive[grid](x, y, output, 
    #                     M, K, N, 
    #                     x.stride(0), x.stride(1),
    #                     y.stride(0), y.stride(1),
    #                     output.stride(0), output.stride(1),
    #                     BLOCK_SIZE = 32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    group_matmul_kernel[grid](x, y, output, 
                        M, K, N, 
                        x.stride(0), x.stride(1),
                        y.stride(0), y.stride(1),
                        output.stride(0), output.stride(1),
                        )
    return output

@triton.jit
def batch_matmul_kernel(
                # my pointers
                x_ptr,y_ptr,output_ptr,
                # matrix dims
                B, H, M, K, N,
                # strides for each matrix
                stride_xb, stride_xh, stride_xm, stride_xk,
                stride_yb, stride_yh, stride_yk, stride_yn,
                stride_ob, stride_oh, stride_om, stride_on,
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, 
                BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr):

    pid_batch = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    # Number of program ids along M dimension
    # num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    # num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # pid_m = pid_mat // N
    # pid_n = pid_mat % N

    # first_pid_m = pid_m * GROUP_SIZE_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # pid_m = first_pid_m + (pid_mat % group_size_m)

    # strides for the M, K and N dim. 
    # A[b, i, j] = b * stride_Ab  + i *stride_Am + j * stride_An
    # &A[b, m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + A.stride(0) + (m : m+BLOCK_SIZE_M)*A.stride(1) + (k : k+BLOCK_SIZE_K)*A.stride(2);
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    x_ptr += pid_b * stride_xb + pid_h * stride_xh + offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += pid_b * stride_yb + pid_h * stride_yh + offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptr, mask = offset_k [None, :] < K - k * BLOCK_SIZE_K, other = 0.0)
        y = tl.load(y_ptr, mask = offset_k[:,None] < K - k * BLOCK_SIZE_K, other = 0.0)            
        accumulator = tl.dot(x, y, accumulator)
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K *  stride_xk
        y_ptr += BLOCK_SIZE_K *  stride_yk
        
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offset = pid_b * stride_ob + pid_h * stride_oh + offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    output_mask =  (offset_om[:,None] < M) & ( offset_on[None, :] < N)
    tl.store(output_ptr + output_offset, accumulator, mask = output_mask)
       
def matmul(x: torch.Tensor, y: torch.Tensor):
    B, H, M , K = x.shape
    B1, H1, K1, N = y.shape
    assert (K == K1) & (B == B1) * (H == H1)
    output = torch.empty((B, H, M, N), device = 'cuda', dtype = y.dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    grid = lambda meta: (B*H, triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    batch_matmul_kernel[grid](x, y, output, 
                        B, H, M, K, N, 
                        *x.stride(), *y.stride(), *output.stride(),
                        BLOCK_SIZE_M = 32, 
                        BLOCK_SIZE_N = 32, 
                        BLOCK_SIZE_K = 32, 
                        # GROUP_SIZE along M
                        GROUP_SIZE_M = 8
                        )
    return output


# dtype = torch.float32
# x = torch.rand((32, 16), device = 'cuda', dtype = dtype)
# y = torch.rand((16, 45), device = 'cuda', dtype = dtype)
# output_torch = torch.matmul(x, y).to(torch.float16)
# output_triton = matmul(x, y, dtype = dtype).to(torch.float16)
# print(f"Are the results close? {torch.allclose(output_triton, output_torch, atol=1e-2, rtol=0)}")
