# Cell 6: WaveSentenceComposer
import torch.nn.functional as F
import torch.fft as fft
from typing import Optional # Import Optional

class WaveSentenceComposer(nn.Module):
    """
    Combine token time wavelets, semantic wave params, and frequency-attended reconstruction.
    forward(token_waves (B,S,T,D), token_params (B,S,D,P), semantic_params (B,D,P)) ->
        composite_waves (B,S,T,D)
    """
    def __init__(self, time_steps: int, num_dimensions: int, fft_attn: Optional[MultiHeadFFTAttention] = None):
        super().__init__()
        self.time_steps = time_steps
        self.num_dimensions = num_dimensions
        # gating networks to combine components per token
        self.gate_net = nn.Sequential(
            nn.Linear(num_dimensions * 3, 128),
            nn.GELU(),
            nn.Linear(128, num_dimensions),
            nn.Sigmoid()
        )
        self.fft_attn = fft_attn  # expected to be MultiHeadFFTAttention instance or None

    def forward(self,
                token_waves: torch.Tensor,
                token_params: torch.Tensor,
                semantic_params: torch.Tensor) -> torch.Tensor:
        """
        token_waves: (B, S, T, D)
        token_params: (B, S, D, P)
        semantic_params: (B, D, P)
        returns:
          composite_waves: (B, S, T, D)
        """
        # print(f"WaveSentenceComposer Input - token_waves.requires_grad: {token_waves.requires_grad}, token_waves.grad_fn: {token_waves.grad_fn}")
        # print(f"WaveSentenceComposer Input - token_params.requires_grad: {token_params.requires_grad}, token_params.grad_fn: {token_params.grad_fn}")
        # print(f"WaveSentenceComposer Input - semantic_params.requires_grad: {semantic_params.requires_grad}, semantic_params.grad_fn: {semantic_params.grad_fn}")


        B, S, T, D = token_waves.shape
        # build semantic time wave from semantic_params: (B, 1, T, D)
        A_sem = semantic_params[..., 0]  # (B,D)
        omega_sem = semantic_params[..., 1]
        phi_sem = semantic_params[..., 2]
        t = make_time_vector(self.time_steps, device=token_waves.device, dtype=token_waves.dtype).view(1,1,T,1)
        sem_time = A_sem.unsqueeze(1).unsqueeze(2) * torch.sin(omega_sem.unsqueeze(1).unsqueeze(2) * t + phi_sem.unsqueeze(1).unsqueeze(2))
        # sem_time: (B,1,T,D) -> broadcast to (B,S,T,D)
        sem_time_b = sem_time.expand(B, S, T, D)

        # freq-attended reconstruction (if provided)
        if self.fft_attn is not None:
            attn_recon = self.fft_attn(token_waves)  # (B,S,T,D)
        else:
            attn_recon = torch.zeros_like(token_waves)

        # gating: per-token, per-dimension weight to combine original token wave and semantic and attended
        # prepare gating input: use token params A, omega, phi averaged across time: take A (B,S,D)
        A_tok = token_params[..., 0]  # (B,S,D)
        w_tok = token_params[..., 1]
        token_amp = A_tok  # (B,S,D)
        sem_amp = semantic_params.unsqueeze(1)[..., 0]  # (B,1,D)
        attn_pow = attn_recon.abs().mean(dim=2)  # (B,S,D)
        gate_features = torch.stack([token_amp, sem_amp.expand(B,S,D), attn_pow], dim=-1)  # (B,S,D,3)
        gate_in_flat = gate_features.view(B*S, D*3)
        gates = self.gate_net(gate_in_flat).view(B, S, D)  # (B,S,D) in [0,1]

        # combine: composite = g * token_waves + (1-g) * (alpha*sem + beta*attn)
        # expand gates to time axis
        g_time = gates.unsqueeze(2)  # (B,S,1,D)
        composite = g_time * token_waves + (1.0 - g_time) * (0.5 * sem_time_b + 0.5 * attn_recon)

        # print(f"WaveSentenceComposer Output - composite.requires_grad: {composite.requires_grad}, composite.grad_fn: {composite.grad_fn}")

        return composite