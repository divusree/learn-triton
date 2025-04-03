import triton
import triton.language as tl
import torch
import math
torch.set_printoptions(profile="full")
@triton.jit
def fa_fwd_inner_kernel(q_ptr, k_ptr, v_ptr, output_ptr, L_ptr, qk_scale,
                B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                stride_qb, stride_qh, stride_qm, stride_qn, # all of the matrices have the same stride
                stride_ob, stride_oh, stride_om, stride_on, 
                stride_Lb, stride_Lh, stride_Lm, stride_Ln, 
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr 
            ):
    pid_batch = tl.program_id(axis=0) 
    pid_row = tl.program_id(axis=1)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    offset_row_block = (pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_col = tl.arange(0, BLOCK_SIZE_N) % BLOCK_SIZE_N
    
    offset = pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col[None,:] *stride_qn
    q_ptr += offset
    mask = (offset_row_block[:,None] < M) & (offset_col[None, :] < N)
    Q = tl.load(q_ptr, mask = mask, other = 0)
    k_offset = pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col[None,:] *stride_qn
    k_ptr += k_offset # transposed K tensor by swapping the strides of the last 2 dimensions
    v_ptr += offset

    m = tl.full((BLOCK_SIZE_M, 1), value = -torch.inf, dtype = tl.float32) # axis = 1 gives me row wise values
    l = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32)    
    O = tl.zeros((BLOCK_SIZE_M, N), dtype = tl.float32)

    for j in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        K = tl.load(k_ptr, mask = (offset_row_block[:,None] < M - j *BLOCK_SIZE_M), other = 0) 
        V = tl.load(v_ptr, mask = (offset_row_block[:,None] < M - j *BLOCK_SIZE_M), other = 0) 
        S = tl.dot(Q,K) * qk_scale
        prev_m = m
        m = tl.maximum(tl.max(S, axis = -1, keep_dims = True), prev_m)
        P = tl.exp(S - m)
        corr = tl.exp(prev_m - m)
        l = corr * l + tl.sum(P, axis = -1, keep_dims = True) 
        O = (corr) * O + tl.dot(P, V, input_precision = 'ieee')
        k_ptr += BLOCK_SIZE_M * stride_qn
        v_ptr += BLOCK_SIZE_M * stride_qm

    O = (1/l) * O
    L = m + tl.log(l)
    output_offset =  pid_b * stride_ob + pid_h * stride_oh + offset_row_block[:,None] * stride_om + offset_col[None,:] *stride_on
    tl.store(output_ptr + output_offset, O) 
    L_offset =  pid_b * stride_Lb + pid_h * stride_Lh + offset_row_block[:,None] * stride_Lm + offset_col[None,:] *stride_Ln
    tl.store(L_ptr + L_offset, L)


def fa_fwd_inner(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    B, H, M, N = Q.shape
    output = torch.zeros_like(Q, device = 'cuda', dtype = torch.float32)
    L = torch.zeros((B, H, M, 1), device = 'cuda', dtype = torch.float32)
    qk_scale = math.sqrt(N)
    assert Q.is_cuda and K.is_cuda and V.is_cuda 
    grid = lambda meta: (B*H, triton.cdiv(M, meta['BLOCK_SIZE_M']))
    fa_fwd_inner_kernel[grid](Q, K, V, output, L, qk_scale,
                    B, H, M, N,   
                    *Q.stride(), 
                    *output.stride(), 
                    *L.stride(), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = triton.next_power_of_2(N))   
  
    return output, L


@triton.jit
def fa_bwd_inner_kernel(q_ptr, k_ptr, v_ptr,
                        o_ptr, dO_ptr, L_ptr, qk_scale,
                        dQ_ptr, dK_ptr, dV_ptr, D_ptr, 
                        M: tl.constexpr , N: tl.constexpr, 
                        stride_qm, stride_qn, # all of the matrices have the same stride
                        stride_lm, stride_ln, 
                        BLOCK_SIZE_M: tl.constexpr, 
                        BLOCK_SIZE_N: tl.constexpr 
                        ):


    pid_row = tl.program_id(axis = 0)
    pid_col = tl.program_id(axis = 1)

 
    offset_kv_row_block = (pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_kv_col_block = (pid_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % BLOCK_SIZE_N # check the modulo

    offset = offset_kv_row_block[:,None] * stride_qm + offset_kv_col_block[None,:] *stride_qn
    v_ptr += offset
    k_ptr += offset
    kv_mask = (offset_kv_row_block[:,None] < M) & (offset_kv_col_block[None, :] < N)
    K_j = tl.load(k_ptr, mask = kv_mask, other = 0)
    V_j = tl.load(v_ptr, mask = kv_mask, other = 0)

    offset_row_block =  tl.arange(0, BLOCK_SIZE_M)% M
    offset_col_block = tl.arange(0, BLOCK_SIZE_N) % BLOCK_SIZE_N # check the modulo

    q_ptr += offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    o_ptr += offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    dO_ptr += offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    dQ_ptr += offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    q_mask  = (offset_row_block[:,None] < M) & (offset_col_block[None, :] < N)
    dQ_offset = offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn

    L_ptr += offset_row_block[:,None] * stride_lm #+offset_col_block[None,:] *stride_ln
    D_ptr += offset_row_block[:,None] * stride_lm #+offset_col_block[None,:] *stride_ln
    L_mask =(offset_row_block[:,None] < M) #and (offset_col_block[None,:] < N) 

    dV_j = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    dK_j = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)

    for i in range(0, tl.cdiv(M, BLOCK_SIZE_M)):

        
        Q_i = tl.load(q_ptr, mask = q_mask, other = 0) # check mask
        # O_i = tl.load(o_ptr, mask = q_mask, other = 0) # check mask
        dO_i = tl.load(dO_ptr, mask = q_mask, other = 0) # check mask
        dQ_i = tl.load(dQ_ptr, mask = q_mask, other = 0) # check mask
        L_i = tl.load(L_ptr, mask = L_mask, other = 0) 
        D_i = tl.load(D_ptr, mask = L_mask, other = 0) 

        # back pass
        S_i = tl.dot(Q_i, K_j.trans(1,0)) * qk_scale
        P_i = tl.exp(S_i - L_i)
        # if ((pid_row == 0) and (pid_col == 0)) and (i == 0):
        #     tl.device_print("S_i - L_i", S_i - L_i)
        dV_j += tl.dot(P_i.T, dO_i)
        dP_i = tl.dot(dO_i, V_j.T)
        dS_i = P_i * (dP_i - D_i)
        dQ_i += tl.dot(dS_i, K_j) # write back to HBM
        dK_j += tl.dot(dS_i.T, Q_i)

        tl.store(dQ_ptr + dQ_offset, dQ_i, mask = q_mask)

        q_ptr += BLOCK_SIZE_M * stride_qm
        dO_ptr += BLOCK_SIZE_M * stride_qm
        dQ_ptr += BLOCK_SIZE_M * stride_qm
        L_ptr += BLOCK_SIZE_M * stride_lm
        D_ptr += BLOCK_SIZE_M * stride_lm

    tl.store(dK_ptr + offset, dK_j, mask = kv_mask)
    tl.store(dV_ptr + offset, dV_j, mask = kv_mask)


def fa_bwd_inner(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                 O: torch.Tensor, dO: torch.Tensor,
                 L: torch.Tensor):
    """
    inputs - dO from torch - upstream gradient 

    outputs - dQ, dK, dV - downstream gradient. helper vars - dS, dP
    """
    # B, M, N = Q.shape
    assert Q.is_cuda and K.is_cuda and V.is_cuda and O.is_cuda and dO.is_cuda and L.is_cuda

    M, N = Q.shape
    qk_scale = 1/ math.sqrt(N)

    dQ = torch.zeros_like(Q, device = 'cuda', dtype = torch.float32)
    dK = torch.zeros_like(K, device = 'cuda', dtype = torch.float32)
    dV = torch.zeros_like(V, device = 'cuda', dtype = torch.float32)

    D = O * dO # optimize?
    D = D.sum(dim = 1).reshape(M, 1)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), 1)
    fa_bwd_inner_kernel[grid](
                            Q, K, V, 
                            O, dO, L, qk_scale,
                            dQ, dK, dV, D,
                            M, N,      
                            Q.stride(0), Q.stride(1), 
                            L.stride(0), L.stride(1), 
                            BLOCK_SIZE_M = 16,
                            BLOCK_SIZE_N = triton.next_power_of_2(N))   
  
    return dQ, dK, dV