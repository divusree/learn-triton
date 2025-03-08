import triton
import triton.language as tl
import torch
print("is cuda being used?", torch.cuda.is_available())

@triton.jit
def matmul_kernel(x_ptr,y_ptr,output_ptr,n,BLOCK_SIZE: tl.constexpr):
    # these values are the same for both x and y
    pid = tl.program_id(axis = 0) # row vector pids
    block_start = pid * BLOCK_SIZE
    stride = tl.arange(0, BLOCK_SIZE) 
    offset = block_start * pid + stride

    mask = offset < n

    x = tl.load(x_ptr + offset, mask = mask)
    y = tl.load(y_ptr + offset, mask = mask)

    output = x * y
    output = tl.sum(output, axis = 0)
    tl.atomic_add(output_ptr, output) # no mask!    

# vector dot product. output should be 1x1 tensor
def matmul(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty((x.shape[0], y.shape[1]), device = 'cuda')
    assert x.shape[1] == y.shape[0]
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']), )
    matmul_kernel[grid](x, y, output, n, BLOCK_SIZE = 32)
    return output
