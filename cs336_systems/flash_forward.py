import math

import torch
import triton
import triton.language as tl

from cs336_systems.benchmark_attention import attention

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(query_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(Q_TILE_SIZE, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(Q_TILE_SIZE,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    ) 

    O = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    M = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    for i in range(0, N_KEYS, K_TILE_SIZE):
        Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        K = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    
        M_old = tl.full((Q_TILE_SIZE,), 1.0, dtype=tl.float32)
        M_old = M_old * M

        S_i_j = tl.dot(Q, K.trans(), input_precision="ieee") / scale 
        rowwise_max = tl.max(S_i_j, axis=-1)
        M = tl.where(M_old > rowwise_max, M_old, rowwise_max) 
        P = tl.exp(S_i_j - M[:, None])
        P.to(dtype=V.dtype) 
        L = tl.exp(M_old - M) * L + tl.sum(P, axis=-1)
        O = tl.exp(M_old - M)[:, None] * O + tl.dot(P, V, input_precision="ieee")

        Q_block_ptr = Q_block_ptr.advance((0, Q_TILE_SIZE))
        K_block_ptr = K_block_ptr.advance((0, K_TILE_SIZE))
        V_block_ptr = V_block_ptr.advance((0, K_TILE_SIZE))

    O = (1 / L)[:, None] * O
    L = M + tl.log(L)

    tl.store(O_block_ptr, O, boundary_check=(0,1))    
    tl.store(L_block_ptr, L, boundary_check=(0,))    

class FlashForwardTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size = Q.shape[0]

        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        d_q = Q.shape[-1]
        d_k = K.shape[-1]

        b_q = 16
        b_k = 16

        t_q = triton.cdiv(n_q, b_q)
        t_k = triton.cdiv(n_k, b_k)

        O = torch.zeros((batch_size, n_q, d_q), device=Q.device, dtype=torch.float32)
        L = torch.zeros((batch_size, n_q,), device=Q.device, dtype=torch.float32)

        flash_fwd_kernel[(t_q, batch_size)](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=n_q, N_KEYS=n_k,
            scale=math.sqrt(d_k), D=d_k,
            Q_TILE_SIZE=b_q, K_TILE_SIZE=b_k,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_outputs):
        raise NotImplementedError


class FlashForwardTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size = Q.shape[0]

        n_q = Q.shape[-2]
        n_k = K.shape[-2]
        d_k = K.shape[-1]

        b_q = 16
        b_k = 16

        t_q = math.ceil(n_q / b_q) 
        t_k = math.ceil(n_k / b_k) 

        O = torch.zeros((batch_size, n_q, d_k), device=Q.device)
        L = torch.zeros((batch_size, n_q, ), device=Q.device)
        for b in range(batch_size):
            for i in range(0, t_q):
                offsets = i * b_q + torch.arange(0, b_q)
                offsets = offsets[offsets < n_q]
                Q_i = Q[b, offsets, :]
                O_i = O[b, offsets, :]
                l_i = torch.zeros((Q_i.shape[-2],), device=Q.device)
                m_i = torch.zeros((Q_i.shape[-2],), device=Q.device) + float("-inf")

                for j in range(0, t_k):
                    j_offsets = j * b_k + torch.arange(0, b_k)
                    j_offsets = j_offsets[j_offsets < n_k]
                    K_j = K[b, j_offsets, :]
                    V_j = V[b, j_offsets, :]

                    old_m = m_i.clone()
                    old_l = l_i.clone()
                    old_o = O_i.clone() 

                    # (b_q, b_k)
                    S_i_j = (Q_i @ K_j.permute(1, 0)) / math.sqrt(d_k)

                    # (b_q)
                    rowwise_max = torch.max(S_i_j, dim=-1)[0]

                    # (b_q) 
                    m_i = torch.where(old_m > rowwise_max, old_m, rowwise_max) 

                    # (b_q, b_k) 
                    P = torch.exp(S_i_j - m_i.unsqueeze(-1))

                    # (b_q) 
                    l_i = torch.exp(old_m - m_i) * old_l + torch.sum(P, dim=-1)

                    # (b_q, d)
                    O_i = torch.diag(torch.exp(old_m - m_i)) @ old_o + P @ V_j 

                O[b, offsets, :] = torch.diag(1 / l_i) @ O_i
                L[b, offsets] = m_i + torch.log(l_i) 

        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_outputs):
        raise NotImplementedError

if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 2
    seq_length = 16 
    d_model = 16 

    Q = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)
    K = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)
    V = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)

    torch_attn = attention(Q, K, V) 
    ours = FlashForwardTorch.apply(Q, K, V)
    ours_triton = FlashForwardTriton.apply(Q, K, V)

    print("Pytorch impl. matches our pytorch version of flashattention?: ", torch.isclose(torch_attn, ours, rtol=1e-7, atol=1e-6)._is_all_true())
    print("Pytorch impl. matches our triton version of flashattention?: ", torch.isclose(torch_attn, ours_triton, rtol=1e-7, atol=1e-6)._is_all_true())


