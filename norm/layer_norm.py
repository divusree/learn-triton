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
                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr
                ):

    pid_m = tl.program_id(axis = 0) 
    pid_n = tl.program_id(axis = 1) 
    offset = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_xm +  (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None,:]*stride_xn
    x_ptr += offset
    mean = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32 )
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n + (tl.arange(0, BLOCK_SIZE_N))[None, :] 
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)   
        mean += tl.sum(x, axis = -1, keep_dims = True) 
    mean = tl.sum(mean, axis = -1, keep_dims = True) /M
    
    variance = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32 )
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n + (tl.arange(0, BLOCK_SIZE_N))[None, :] 
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)    
        variance += tl.sum((x- mean)*(x- mean), axis = -1, keep_dims = True)

    variance = tl.sum(variance, axis = -1, keep_dims = True) /N
    rstd = 1/tl.sqrt(variance + eps)

    output_ptr += offset
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n + (tl.arange(0, BLOCK_SIZE_N))[None, :] 
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)   
        output = (x - mean) * rstd        
        tl.store(output_ptr  + offset_col ,  output, mask = offset_col < N)

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
                    BLOCK_SIZE_N = 32
                    )
    return output


@triton.jit
def batched_layer_norm_kernel(x_ptr,output_ptr, eps,
                B, H, M, N: tl.constexpr,
                stride_xb, stride_xh, stride_xm, stride_xn,
                mean_ptr, var_ptr,
                stride_mb, stride_mh, stride_mm, stride_mn,

                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr,
                ):
    # # calculate per batch
    # # layer norm - happens over the columns/features of the tensor. we are normalising the features
    # # output = gamma * (x_i - mu)/ sqrt(sig**2 + eps) + beta
    # mean = torch.empty((M, 1))
    # variance = torch.empty((M, 1))

    # mean = y.mean(dim = -1, keepdim = True) # reduced across rows
    # variance = torch.square(y - mean).mean(dim = -1, keepdim = True)
    # eps = 1e-8
    # y_custom = (y - mean) / torch.sqrt(variance + eps)
    # y_custom

    pid_batch = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    offset = pid_b * stride_xb + pid_h * stride_xh + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_xm +  tl.arange(0, BLOCK_SIZE_N)[None,:]*stride_xn
    x_ptr += offset
    mean = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32 )
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n * BLOCK_SIZE_N * stride_xn
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)   
        mean += tl.sum(x, axis = -1, keep_dims = True) 
    mean = tl.sum(mean, axis = -1, keep_dims = True) /N
    tl.store(mean_ptr + pid_b * stride_mb + pid_h * stride_mh +  (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_mm , mean)

    variance = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32 )
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n * BLOCK_SIZE_N * stride_xn
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)    
        variance += tl.sum((x- mean)*(x- mean), axis = -1, keep_dims = True)

    variance = tl.sum(variance, axis = -1, keep_dims = True) /N
    rstd = 1/tl.sqrt(variance + eps)
    tl.store(var_ptr + pid_b * stride_mb + pid_h * stride_mh +  (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_mm, variance)

    output_ptr += offset
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n * BLOCK_SIZE_N * stride_xn
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)   
        output = (x - mean) * rstd        
        tl.store(output_ptr  + offset_col ,  output, mask = offset_col < N)

def batched_layer_norm(x: torch.Tensor):
    B, H, M , N = x.shape
    output = torch.zeros_like(x,  device = x.device, dtype = x.dtype)
    assert x.is_cuda and output.is_cuda
    eps = 1e-6
    mean = torch.zeros((B, H, M, 1), device = x.device, dtype = x.dtype)
    var = torch.zeros((B, H, M, 1), device = x.device, dtype = x.dtype)
    grid = lambda meta: (B * H, triton.cdiv(M, meta['BLOCK_SIZE_M']))
    batched_layer_norm_kernel[grid](x, output, eps,
                    B, H, M , N,
                    *x.stride(),
                    mean, var, *mean.stride(),
                    BLOCK_SIZE_M = 32, 
                    BLOCK_SIZE_N = 32
                    )
    return output, mean, var