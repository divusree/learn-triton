import triton
import triton.language as tl

import torch
print("is cuda being used?", torch.cuda.is_available())

@triton.jit
def addition1D_kernel(
        x_ptr, 
        y_ptr,
        output_ptr,
        n,
        BLOCK_SIZE: tl.constexpr,
        ):
    pid = tl.program_id(axis = 0) 
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask = mask)

@triton.jit
def addition2D_kernel(
        x_ptr, 
        y_ptr,
        output_ptr,
        nx,
        ny, 
        BLOCK_SIZE: tl.constexpr, 
        ):
    pidx = tl.program_id(axis = 0)
    block_startx = pidx * BLOCK_SIZE

    pidy = tl.program_id(axis = 1)
    block_starty = pidy * BLOCK_SIZE

    offsetsx = block_startx + tl.arange(0, BLOCK_SIZE) 
    offsetsy = block_starty + tl.arange(0, BLOCK_SIZE) 


    offsets = offsetsy * ny + offsetsx
    mask = (offsets < nx*ny)

    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask = mask)

def add_1D(x:torch.Tensor, y:torch.Tensor):
    output = torch.empty_like(y)
    assert x.is_cuda and y.is_cuda and output.is_cuda 
    n = output.numel()

    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), ) 

    addition1D_kernel[grid](x, y, output, n, BLOCK_SIZE = 1024)
    return output

def add_2D(x:torch.Tensor, y:torch.Tensor):
    output = torch.empty_like(y)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    nx, ny = output.shape
    grid = lambda meta: (triton.cdiv(nx, meta['BLOCK_SIZE']), triton.cdiv(ny, meta['BLOCK_SIZE'])) 
    addition2D_kernel[grid](x, y, output, nx, ny, BLOCK_SIZE = 32)
    return output

@triton.jit
def addition3D_kernel(
        x_ptr, #ptr to first idx of x, just like in cpp
        y_ptr,
        output_ptr,
        B, M, N, 
        stride_xb,stride_xm, stride_xn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        GROUP_SIZE_M: tl.constexpr 
        ):
    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset =  pid_b * stride_xb + offset_xm[:, None] * stride_xm + offset_xn[None, :] * stride_xn
    mask = (offset_xm[:, None] < M) & (offset_xn[None, :] < N)
    x = tl.load(x_ptr + offset, mask = mask, other = 0.0)
    y = tl.load(y_ptr + offset, mask = mask, other = 0.0)
    tl.store(output_ptr + offset, x + y, mask = mask)
def add_3D(x:torch.Tensor, y:torch.Tensor):
    # preallocate output
    assert x.shape == y.shape
    output = torch.empty_like(x, device = x.device, dtype = x.dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    B, M, N = output.shape

    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N'])) 
    addition3D_kernel[grid](x, y, output, 
                                  B, M, N, 
                                  *x.stride(),
                                  BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 32,
                                  GROUP_SIZE_M = 8)
    return output   