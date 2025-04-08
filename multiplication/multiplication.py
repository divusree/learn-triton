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
def matmul_naive(x: torch.Tensor, y: torch.Tensor):
    M , K = x.shape
    K1, N = y.shape
    # assert K == K1
    output = torch.empty((M, N), device = 'cuda', dtype = torch.float32)
    assert x.is_cuda and y.is_cuda and output.is_cuda

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    matmul_kernel_naive[grid](x, y, output, 
                        M, K, N, 
                        x.stride(0), x.stride(1),
                        y.stride(0), y.stride(1),
                        output.stride(0), output.stride(1),
                        BLOCK_SIZE_N = 32, 
                        BLOCK_SIZE_K = 32, 
                        BLOCK_SIZE_M = 32, 
                        )
    return output

# @triton.autotune(
#     configs=get_autotune_config(),
#     key=['M', 'N', 'K'])
@triton.jit
def matmul3D_kernel(
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


    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid = pid_m #keep this as is
   
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # strides for the M, K and N dim. 
    # A[b, i, j] = b * stride_Ab  + i *stride_Am + j * stride_An
    # &A[b, m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + A.stride(0) + (m : m+BLOCK_SIZE_M)*A.stride(1) + (k : k+BLOCK_SIZE_K)*A.stride(2);
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    x_ptr += pid_b * stride_xb  + offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += pid_b * stride_yb  + offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptr, mask= offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.load(y_ptr, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)       
        accumulator += tl.dot(x, y, input_precision = 'ieee')
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K *  stride_xk
        y_ptr += BLOCK_SIZE_K *  stride_yk
        
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offset = pid_b * stride_ob  + offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    output_mask =  (offset_om[:,None] < M) & ( offset_on[None, :] < N)
    tl.store(output_ptr + output_offset, accumulator, mask = output_mask)
    
def matmul3D(x: torch.Tensor, y: torch.Tensor, xtrans = False, ytrans = False):
    B, M , K = x.shape
    B1, L, N = y.shape
    assert (B == B1)
    stride_xb, stride_xm, stride_xk = x.stride()
    stride_yb, stride_yk, stride_yn = y.stride()
    output = torch.empty((B, M, N), device = x.device, dtype = x.dtype)
    if xtrans:
        assert M == L 
        stride_xm, stride_xk = stride_xk, stride_xm
        output = torch.empty((B, K, N), device = x.device, dtype = x.dtype)
        M, K = K, M # check this
    elif ytrans:
        assert N == K
        stride_yk, stride_yn = stride_yn, stride_yk
        output = torch.empty((B, M, L), device = x.device, dtype = x.dtype)
        N, L = L, N # check this
    elif (not xtrans) and (not ytrans):
        assert K == L  
    else:
        raise NotImplementedError("x.T @ y.T is not implemented")
    assert x.is_cuda and y.is_cuda and output.is_cuda
    BLOCK_SIZE_M = 64 # min(max(16, M), 64)
    BLOCK_SIZE_N = 64 # min(max(16, N), 64)
    BLOCK_SIZE_K = 32  # Keep K block size smaller for register pressure
    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M'])*triton.cdiv(N, meta['BLOCK_SIZE_N']))
    matmul3D_kernel[grid](x, y, output, 
                        B, M, K, N, 
                        stride_xb, stride_xm, stride_xk,
                        stride_yb, stride_yk, stride_yn,
                        *output.stride(),
                        BLOCK_SIZE_B = 1, BLOCK_SIZE_M = BLOCK_SIZE_M, BLOCK_SIZE_N = BLOCK_SIZE_N, BLOCK_SIZE_K = BLOCK_SIZE_K, GROUP_SIZE_M = 8
                        )
    return output
@triton.jit
def matmul4D_kernel(
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
    pid = pid_m
    pid_b = pid_batch // H
    pid_h = pid_batch % H
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # strides for the M, K and N dim. 
    # A[b, i, j] = b * stride_Ab  + i *stride_Am + j * stride_An
    # &A[b, m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + A.stride(0) + (m : m+BLOCK_SIZE_M)*A.stride(1) + (k : k+BLOCK_SIZE_K)*A.stride(2);
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    x_ptr += pid_b * stride_xb + pid_h * stride_xh + offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += pid_b * stride_yb + pid_h * stride_yh + offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptr, mask= offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.load(y_ptr, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)       
        accumulator += tl.dot(x, y, input_precision = 'ieee')
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K *  stride_xk
        y_ptr += BLOCK_SIZE_K *  stride_yk
        
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offset = pid_b * stride_ob + pid_h * stride_oh + offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    output_mask =  (offset_om[:,None] < M) & ( offset_on[None, :] < N)
    tl.store(output_ptr + output_offset, accumulator, mask = output_mask)

def matmul4D(x: torch.Tensor, y: torch.Tensor, xtrans = False, ytrans = False):
    # Check dimensions
    B, H, M, K = x.shape
    B1, H1, L, N = y.shape
    assert (B == B1) and (H == H1)
    # Allocate output
    output = torch.empty((B, H, M, N), device='cuda', dtype=y.dtype)
    BLOCK_SIZE_M = 64 # min(max(16, M), 64)
    BLOCK_SIZE_N = 64 # min(max(16, N), 64)
    BLOCK_SIZE_K = 32  # Keep K block size smaller for register pressure
    output = torch.empty((B, H, M, N), device = x.device, dtype = x.dtype)
    
    stride_xb,stride_xh, stride_xm, stride_xk = x.stride()
    stride_yb, stride_yh, stride_yk, stride_yn = y.stride()
    if xtrans:
        assert M == L 
        stride_xm, stride_xk = stride_xk, stride_xm
        output = torch.empty((B, H, K, N), device = x.device, dtype = x.dtype)
        M,K = K, M # check this
    elif ytrans:
        assert N == K
        stride_yk, stride_yn = stride_yn, stride_yk
        output = torch.empty((B, H, M, L), device = x.device, dtype = x.dtype)
        N, L = L, N # check this
    elif (not xtrans) and (not ytrans):
        assert K == L  
    else:
        raise NotImplementedError("x.T @ y.T is not implemented")
    # Grid computation
    grid = lambda _: (B * H, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    matmul4D_kernel[grid](
        x, y, output,
        B, H, M, K, N,
        stride_xb, stride_xh, stride_xm, stride_xk,
        stride_yb, stride_yh, stride_yk, stride_yn,
        *output.stride(),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8
    )
    return output

def matmul(x,y, xtrans = False, ytrans = False):
    if (len(x.shape) == 3) and (len(y.shape) == 3):
        return matmul3D(x, y, xtrans, ytrans)
    elif (len(x.shape) == 4) and (len(y.shape) == 4):
        return matmul4D(x, y, xtrans, ytrans)
    elif (len(x.shape) == 2) and (len(y.shape) == 2):
        return matmul3D(x.unsqueeze(0), y.unsqueeze(0), xtrans, ytrans).squeeze(0)
    elif (len(x.shape) == 3) and (len(y.shape) == 2):
        return matmul3D(x, y.unsqueeze(0), xtrans, ytrans).squeeze(0)
    elif (len(x.shape) == 2) and (len(y.shape) == 3):
        return matmul3D(x.unsqueeze(0), y, xtrans, ytrans).squeeze(0)
    else:
        raise NotImplementedError(f"x.shape and y.shape incompatible {x.shape, y.shape}")        
