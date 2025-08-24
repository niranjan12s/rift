# Cell 5: MultiHeadFFTAttention
import torch.nn.functional as F
import torch.fft as fft

class MultiHeadFFTAttention(nn.Module):
    def __init__(self, time_steps: int, num_dimensions: int, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.time_steps = time_steps
        self.num_dimensions = num_dimensions
        self.num_heads = num_heads

        # frequency bins count from rfft
        self.freq_bins = time_steps // 2 + 1

        # We'll operate per-dimension by flattening D into the freq vector:
        self.freq_repr_dim = self.freq_bins * num_dimensions

        # projection to multi-head (Q/K/V same proj here)
        # produce (num_heads * head_dim)
        self.to_heads = nn.Linear(self.freq_repr_dim, num_heads * head_dim)
        self.head_dim = head_dim

        # output projection from concatenated heads -> freq_repr_dim
        self.out_proj = nn.Linear(num_heads * head_dim, self.freq_repr_dim)

    def forward(self, time_waves: torch.Tensor) -> torch.Tensor:
        """
        time_waves: (B, S, T, D)
        returns: attended_time_waves (B, S, T, D)
        """
        B, S, T, D = time_waves.shape
        assert T == self.time_steps and D == self.num_dimensions

        # rfft across time -> complex tensor (B, S, F, D)
        freq_complex = fft.rfft(time_waves, n=self.time_steps, dim=2)  # last dim is T -> F
        # get magnitude and phase: magnitude (B,S,F,D), phase (B,S,F,D)
        mag = torch.abs(freq_complex)  # real
        phase = torch.angle(freq_complex)  # real

        # flatten dimensions to a single frequency vector per token: (B, S, F*D)
        mag_flat = mag.permute(0, 1, 3, 2).contiguous().view(B, S, -1)  # (B,S, F*D) using (D,F) ordering

        # project to heads
        head_proj = self.to_heads(mag_flat)  # (B,S, num_heads*head_dim)
        head_proj = head_proj.view(B, S, self.num_heads, self.head_dim)  # (B,S,H,Hdim)

        # Use symmetric Q=K=V (frequency correlation). Compute attention across tokens.
        # For stability, compute attention per head.
        heads = []
        for h in range(self.num_heads):
            hvec = head_proj[:, :, h, :]  # (B,S,head_dim)
            # compute scores across tokens: (B, S, S)
            scores = torch.matmul(hvec, hvec.transpose(1, 2)) / math.sqrt(self.head_dim)
            attn = torch.softmax(scores, dim=-1)  # (B, S, S)
            # weighted sum of values (here values are also hvec)
            ctx = torch.matmul(attn, hvec)  # (B, S, head_dim)
            heads.append(ctx)

        # concat heads -> (B, S, num_heads*head_dim)
        ctx_cat = torch.cat(heads, dim=-1)
        # project back to freq_repr_dim
        freq_repr_pred = self.out_proj(ctx_cat)  # (B, S, F*D)
        freq_repr_pred = freq_repr_pred.view(B, S, D, self.freq_bins).permute(0,1,3,2).contiguous()  # (B,S,F,D)

        # freq magnitude predicted (non-negative). Ensure positivity with softplus
        new_mag = F.softplus(freq_repr_pred)  # (B,S,F,D)

        # reconstruct complex frequency using original phase
        # use torch.polar for complex construction: polar(magnitude, angle)
        new_complex = torch.polar(new_mag, phase)  # complex tensor shape (B,S,F,D)

        # irfft to time domain -> real (B,S,T,D)
        time_recon = fft.irfft(new_complex, n=self.time_steps, dim=2)

        return time_recon
