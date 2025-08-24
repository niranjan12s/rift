# Cell 4: SemanticWaveGenerator
class SemanticWaveGenerator(nn.Module):
    """
    Produces a per-sentence semantic wave parameter tensor from token-level params.
    Input: token_params (B, S, D, P)
    Output: semantic_params (B, D, P)
    """
    def __init__(self, num_dimensions: int, wavelet_params: int, hidden_dim: int = 512):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.wavelet_params = wavelet_params
        in_dim = num_dimensions * wavelet_params
        # small MLP applied to pooled token representation
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, token_params: torch.Tensor) -> torch.Tensor:
        """
        token_params: (B, S, D, P)
        returns: (B, D, P)
        """
        # print(f"SemanticWaveGenerator Input - token_params.requires_grad: {token_params.requires_grad}, token_params.grad_fn: {token_params.grad_fn}")

        # pool tokens along S: mean pooling
        pooled = token_params.mean(dim=1)  # (B, D, P)
        B, D, P = pooled.shape
        x = pooled.view(B, D * P)
        out = self.mlp(x)
        out = out.view(B, D, P)

        # print(f"SemanticWaveGenerator Output - out.requires_grad: {out.requires_grad}, out.grad_fn: {out.grad_fn}")

        return out