# Cell 7: WaveDecomposer
class WaveDecomposer(nn.Module):
    """
    Learned decomposition: composite sentence wave (B, S, T, D) -> estimate token params (B,S,D,P) and semantic params (B,D,P)
    Uses lightweight MLPs on pooled/time-aggregated features.
    """
    def __init__(self, num_tokens: int, num_dimensions: int, wavelet_params: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_dimensions = num_dimensions
        self.wavelet_params = wavelet_params

        # semantic extractor: from aggregated composite -> D*P
        # Input to semantic_mlp is (B, D), so in_dim should be num_dimensions
        self.semantic_mlp = nn.Sequential(
            nn.Linear(num_dimensions, hidden_dim), # Changed in_dim from num_dimensions * wavelet_params to num_dimensions
            nn.GELU(),
            nn.Linear(hidden_dim, num_dimensions * wavelet_params)
        )

        # token extractor: per-token MLP mapping pooled time features -> S*D*P
        # Input to token_mlp is (B, S*D), so in_dim should be num_tokens * num_dimensions
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens * num_dimensions, hidden_dim), # Changed in_dim from num_dimensions * wavelet_params to num_tokens * num_dimensions
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * num_dimensions * wavelet_params)
        )

    def forward(self, composite_wave: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        composite_wave: (B, S, T, D)
        returns:
          predicted_token_params: (B, S, D, P)
          predicted_semantic_params: (B, D, P)
        """
        # print(f"WaveDecomposer Input - composite_wave.requires_grad: {composite_wave.requires_grad}, composite_wave.grad_fn: {composite_wave.grad_fn}")

        B, S, T, D = composite_wave.shape
        # Pool time dimension via mean -> (B,S,D)
        pooled = composite_wave.mean(dim=2)  # (B,S,D)
        # also make an overall sentence aggregate -> mean over tokens -> (B,D)
        sent_agg = pooled.mean(dim=1)  # (B,D)

        # Predict semantic params from sentence aggregate
        sem_in = sent_agg.view(B, D)  # (B,D)
        sem_flat = self.semantic_mlp(sem_in)  # (B, D*P)
        sem_params = sem_flat.view(B, D, self.wavelet_params)  # (B,D,P)

        # Predict token params: use pooled (B,S,D) -> flatten per sample
        token_in = pooled.view(B, -1)  # (B, S*D)
        # Map to S*D*P
        token_flat = self.token_mlp(token_in)  # (B, S*D*P)
        token_params = token_flat.view(B, S, D, self.wavelet_params)

        # print(f"WaveDecomposer Output - token_params.requires_grad: {token_params.requires_grad}, token_params.grad_fn: {token_params.grad_fn}")
        # print(f"WaveDecomposer Output - sem_params.requires_grad: {sem_params.requires_grad}, sem_params.grad_fn: {sem_params.grad_fn}")


        return token_params, sem_params