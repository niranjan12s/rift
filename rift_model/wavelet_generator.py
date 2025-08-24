# Cell 3: WaveletGenerator

# small helper to create time axis tensor
def make_time_vector(time_steps: int, device=None, dtype=torch.get_default_dtype()):
    # returns shape (time_steps,) in radians [0, 2*pi)
    t = torch.linspace(0, 2 * math.pi, steps=time_steps, device=device, dtype=dtype)
    return t

class WaveletGenerator(nn.Module):
    """
    Maps token ids -> per-token, per-dimension wave parameters (A, omega, phi)
    and produces time-series wavelets.
    API:
      forward(input_ids) -> (wave: (B, S, T, D), params: (B, S, D, P))
    where P = wavelet_params (default 3: amplitude, frequency, phase)
    """
    def __init__(self,
                 vocab_size: int,
                 num_dimensions: int = 32,
                 wavelet_params: int = 3,
                 time_steps: int = 32,
                 freq_scale: float = 1.0,
                 embed_scale: float = 0.01):
        super().__init__()
        assert wavelet_params >= 3
        self.vocab_size = vocab_size
        self.num_dimensions = num_dimensions
        self.wavelet_params = wavelet_params
        self.time_steps = time_steps
        self.freq_scale = freq_scale

        # embedding dimension packs D * P
        embed_dim = num_dimensions * wavelet_params
        self.param_embedding = nn.Embedding(vocab_size, embed_dim)
        # small MLP to map raw embedding to stable param ranges
        self.param_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # init (small)
        nn.init.normal_(self.param_embedding.weight, mean=0.0, std=embed_scale)

        # time vector will be generated at forward (on device)
        # registers not necessary; computed per forward

    def forward(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: (B, S) long
        returns:
          wave: (B, S, T, D)  # real-valued time series
          params: (B, S, D, P) # raw param outputs (A, omega, phi, ...)
        """
        device = input_ids.device
        B, S = input_ids.shape
        embed = self.param_embedding(input_ids)  # (B, S, D*P)
        embed = self.param_mlp(embed)            # (B, S, D*P)
        params = embed.view(B, S, self.num_dimensions, self.wavelet_params)  # (B,S,D,P)

        # Interpret channels: we reserve indices 0 -> A_raw, 1 -> w_raw, 2 -> phi_raw, rest -> extras
        A_raw = params[..., 0]   # (B,S,D)
        w_raw = params[..., 1]
        phi_raw = params[..., 2]

        # map to stable ranges:
        # amplitude positive -> softplus
        A = F.softplus(A_raw) + 1e-6
        # frequency positive -> softplus scaled by freq_scale
        omega = F.softplus(w_raw) * (self.freq_scale)
        # phase -> map via tanh * pi
        phi = torch.tanh(phi_raw) * math.pi

        # stack back into params_out
        params_out = torch.stack([A, omega, phi], dim=-1)  # (B,S,D,3)

        # generate time axis and compute sine wave
        t = make_time_vector(self.time_steps, device=device, dtype=params_out.dtype)  # (T,)
        # reshape for broadcasting: (1,1,T,1)
        t_bs = t.view(1, 1, self.time_steps, 1)

        # compute (B,S,T,D): A * sin(omega * t + phi)
        # need omega broadcast to (B,S,1,D)

        sin_arg = omega.unsqueeze(2) * t_bs + phi.unsqueeze(2)
        sin_wave = torch.sin(sin_arg)
        wave = A.unsqueeze(2) * sin_wave

        # print(f"WaveletGenerator Output - wave.requires_grad: {wave.requires_grad}, wave.grad_fn: {wave.grad_fn}")
        # print(f"WaveletGenerator Output - params_out.requires_grad: {params_out.requires_grad}, params_out.grad_fn: {params_out.grad_fn}")

        return wave, params_out