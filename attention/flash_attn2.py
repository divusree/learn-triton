import triton
import triton.language as tl
import torch
import math
@triton.jit
def fa_kernel(q_ptr, k_ptr, v_ptr, output_ptr, L_ptr, 
                M: tl.constexpr , N: tl.constexpr, 
                stride_qm, stride_qn, # all of the matrices have the same stride
                stride_om, stride_on, 
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr 
            ):
    pid_row = tl.program_id(axis = 0)
    pid_col = tl.program_id(axis = 1)
    offset_row = (pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_col = (pid_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset =  offset_row[:,None] * stride_qm + offset_col[None,:] *stride_qn
    q_ptr += offset
    mask = (offset_row[:,None] < M) & (offset_row[None, :] < N)
    Q = tl.load(q_ptr, mask = mask, other = 0)
    # K is being transposed so we have permuted stride
    k_ptr += offset_row[:,None] * stride_qn + offset_col[None,:] *stride_qm
    # V is loaded the same as Q and will be advanced like K in the for loop
    v_ptr += offset_row[:,None] * stride_qm + (pid_col * N + tl.arange(0, N))[None,:] *stride_qn
    
    m = tl.full((BLOCK_SIZE_M, 1), value = -torch.inf, dtype = tl.float32) # axis = 1 gives me row wise values
    l = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32)    
    O = tl.zeros((BLOCK_SIZE_M, N), dtype = tl.float32)

    for j in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        K = tl.load(k_ptr, mask = mask, other = 0) # has to be transposed - check if .T is fine or if stride does the trick
        V = tl.load(v_ptr, mask = (offset_row[:,None] < M) & ((pid_col * N + tl.arange(0, N))[None,:] < N), other = 0) 
        S = Q*K.T / tl.sqrt(float(BLOCK_SIZE_N))
        prev_m = m
        m = tl.maximum(tl.max(S, axis = 1, keep_dims = True), m)
        P = tl.exp(S - m)
        corr = tl.exp(prev_m - m)
        l = corr * l + tl.sum(P, axis = 1, keep_dims = True) 
        O = (1/corr) * O #+ P * V
        k_ptr += BLOCK_SIZE_N * stride_qn
        v_ptr += BLOCK_SIZE_M * stride_qm
        print("V", V.shape[0], V.shape[1])
        print("P", P.shape[0], P.shape[1])
    O = (1/l) * O
    L = m + tl.log(l)
    output_offset = offset_row[:,None] * stride_om + (pid_col * N + tl.arange(0, N))[None,:] *stride_on
    tl.store(output_ptr + output_offset, O, mask = (offset_row[:,None] < M) & ((pid_col * N + tl.arange(0, N))[None,:] < N) )
    tl.store(L_ptr + offset_row[:,None], L, mask = offset_row[:,None] < M)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    # B, M, N = Q.shape
    M, N = Q.shape
    M, N = K.shape
    M, N = V.shape
    output = torch.zeros((M, N), device = 'cuda', dtype = torch.float32)
    L = torch.zeros((M, 1), device = 'cuda', dtype = torch.float32)
    assert Q.is_cuda and K.is_cuda and V.is_cuda 
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    fa_kernel[grid](Q, K, V, output, L,
                    M, N,      
                    Q.stride(0), Q.stride(1), 
                    output.stride(0), output.stride(1), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = 32)   
  
    return output, L

output, L = flash_attention(Q, K, V)
