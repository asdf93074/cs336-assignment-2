import math
import torch

from cs336_systems.benchmark_attention import attention

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
    batch_size = 5
    seq_length = 257 
    d_model = 129

    Q = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)
    K = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)
    V = torch.randn((batch_size, seq_length, d_model), requires_grad=True, device=device)

    torch_attn = attention(Q, K, V) 
    ours = FlashForwardTorch.apply(Q, K, V)

    print(torch.isclose(torch_attn, ours, rtol=1e-7, atol=1e-6)._is_all_true())


