# Cell 8: WaveTokenClassifier
class WaveTokenClassifier(nn.Module):
    """
    Classifier that maps token-level composite wave parameters to token logits.
    Input: token_params (B, S, D, P) OR token_wave (B,S,T,D) -> we accept params for stability
    Output: logits (B, S, vocab_size)
    """
    def __init__(self, num_dimensions: int, wavelet_params: int, vocab_size: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.wavelet_params = wavelet_params
        self.vocab_size = vocab_size
        in_dim = num_dimensions * wavelet_params
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)]
        layers += [nn.Linear(hidden_dim, vocab_size)]
        self.net = nn.Sequential(*layers)

    def forward(self, token_params: torch.Tensor) -> torch.Tensor:
        """
        token_params: (B, S, D, P) or (B, D, P) if S=1
        returns: logits (B, S, vocab_size) or (B, vocab_size) if S=1
        """
        # print(f"WaveTokenClassifier Input - token_params.requires_grad: {token_params.requires_grad}, token_params.grad_fn: {token_params.grad_fn}")

        if token_params.dim() == 3: # Handle (B, D, P) case when S=1
            B, D, P = token_params.shape
            x = token_params.view(B, D * P) # (B, D*P)
            logits = self.net(x)  # (B, V)
            # print(f"WaveTokenClassifier Output (S=1) - logits.requires_grad: {logits.requires_grad}, logits.grad_fn: {logits.grad_fn}")
            return logits # Return (B, V) directly

        # Original logic for (B, S, D, P)
        B, S, D, P = token_params.shape
        x = token_params.view(B * S, D * P) # (B*S, D*P)
        logits = self.net(x)  # (B*S, V)
        logits = logits.view(B, S, self.vocab_size) # (B, S, V)
        # print(f"WaveTokenClassifier Output (S>1) - logits.requires_grad: {logits.requires_grad}, logits.grad_fn: {logits.grad_fn}")
        return logits