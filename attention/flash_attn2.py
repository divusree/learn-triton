import triton
import triton.language as tl
import torch
import math
@triton.jit
def fa_kernel(q_ptr, k_ptr, v_ptr, output_ptr, L_ptr, qk_scale,
                M: tl.constexpr , N: tl.constexpr, 
                stride_qm, stride_qn, # all of the matrices have the same stride
                stride_om, stride_on, 
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr 
            ):
    pid_row = tl.program_id(axis = 0)
    pid_col = tl.program_id(axis = 1)
    offset_row_block = (pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_col_block = (pid_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offset_row = (pid_row * M + tl.arange(0, M)) % M
    offset_col = (pid_col * N + tl.arange(0, N)) % N
    
    offset = offset_row_block[:,None] * stride_qm + offset_col[None,:] *stride_qn
    q_ptr += offset
    mask = (offset_row_block[:,None] < M) & (offset_col[None, :] < N)
    Q = tl.load(q_ptr, mask = mask, other = 0)
    k_ptr += offset
    v_ptr += offset
    P_ptr +=  offset_row_block[:,None] * stride_qm + offset_row_block[None,:] *stride_qn
    # print(P_ptr.shape)
    m = tl.full((BLOCK_SIZE_M, 1), value = -torch.inf, dtype = tl.float32) # axis = 1 gives me row wise values
    l = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32)    
    O = tl.zeros((BLOCK_SIZE_M, N), dtype = tl.float32)

    for j in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        K = tl.load(k_ptr, mask = (offset_row_block[:,None] < M - j *BLOCK_SIZE_M), other = 0) # has to be transposed - check if .T is fine or if stride does the trick
        V = tl.load(v_ptr, mask = (offset_row_block[:,None] < M - j *BLOCK_SIZE_M), other = 0) 
        S = tl.dot(Q,K.trans(1,0)) * qk_scale
        prev_m = m
        m = tl.maximum(tl.max(S, axis = 1, keep_dims = True), prev_m)
        P = tl.exp(S - m)
        corr = tl.exp(prev_m - m)
        l = corr * l + tl.sum(P, axis = 1, keep_dims = True) 
        O = (corr) * O + tl.dot(P, V)
        k_ptr += BLOCK_SIZE_M * stride_qm
        v_ptr += BLOCK_SIZE_M * stride_qm

    O = (1/l) * O
    L = m + tl.log(l)
    output_offset = offset_row_block[:,None] * stride_om + offset_col *stride_on
    tl.store(output_ptr + output_offset, O) #, mask = (offset_row[:,None] < M) & ((pid_col * N + tl.arange(0, N))[None,:] < N) )
    tl.store(L_ptr + offset_row_block[:,None], L)



def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    # B, M, N = Q.shape
    M, N = Q.shape
    M, N = K.shape
    M, N = V.shape
    output = torch.zeros((M, N), device = 'cuda', dtype = torch.float32)
    L = torch.zeros((M, 1), device = 'cuda', dtype = torch.float32)
    qk_scale = math.sqrt(N)
    assert Q.is_cuda and K.is_cuda and V.is_cuda 
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    fa_kernel[grid](Q, K, V, output, L, qk_scale,
                    M, N,      
                    Q.stride(0), Q.stride(1), 
                    output.stride(0), output.stride(1), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = triton.next_power_of_2(N))   
  
    return output, L

