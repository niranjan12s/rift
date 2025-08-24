@torch.no_grad()
def generate_wave_text(prompt: str, checkpoint_path: str, max_new_tokens=SAMPLE_MAX_TOKENS, temp=1.0):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    config = checkpoint.get("config", {})
    vocab_size_ckpt = config.get("vocab_size", tokenizer.vocab_size)
    num_dimensions = config.get("num_dimensions", NUM_DIMENSIONS)
    wavelet_params = config.get("wavelet_params", WAVELET_PARAMS)
    time_steps = config.get("time_steps", TIME_STEPS)
    num_heads = config.get("num_heads", 4)
    head_dim = config.get("head_dim", 64)
    max_seq_len = config.get("max_seq_len", MAX_SEQ_LEN)
    hidden_dim = config.get("hidden_dim", HIDDEN_DIM)
    # Get sampling parameters from config, with defaults
    top_k = config.get("top_k", 50)
    top_p = config.get("top_p", 0.9)
    temperature = config.get("temperature", 1.0)


    fft_attn = MultiHeadFFTAttention(time_steps, num_dimensions, num_heads, head_dim).to(DEVICE)
    try:
        fft_attn.load_state_dict(checkpoint["fft_attn"])
        print("Loaded fft_attn state_dict.")
    except:
        fft_attn = None
        print("FFT attention not found or failed to load.")

    emitter = WaveletGenerator(vocab_size_ckpt, num_dimensions, wavelet_params, time_steps).to(DEVICE)
    semantic_gen = SemanticWaveGenerator(num_dimensions, wavelet_params, hidden_dim).to(DEVICE)
    composer = WaveSentenceComposer(time_steps, num_dimensions, fft_attn).to(DEVICE)
    decomposer = WaveDecomposer(max_seq_len, num_dimensions, wavelet_params, hidden_dim).to(DEVICE)
    classifier = WaveTokenClassifier(num_dimensions, wavelet_params, vocab_size_ckpt, hidden_dim).to(DEVICE)

    emitter.load_state_dict(checkpoint["emitter"])
    semantic_gen.load_state_dict(checkpoint["semantic_gen"])
    composer.load_state_dict(checkpoint["composer"])
    decomposer.load_state_dict(checkpoint["decomposer"])
    classifier.load_state_dict(checkpoint["classifier"])

    emitter.eval(), semantic_gen.eval(), composer.eval(), decomposer.eval(), classifier.eval()
    if fft_attn: fft_attn.eval()

    # Use tokenizer.tokenize instead of encode
    seed_idxs = tokenizer.tokenize(prompt)
    # Truncate seed_idxs to max_seq_len
    seed_idxs = seed_idxs[:max_seq_len]

    cur = list(seed_idxs)
    last_token = None
    repeat_count = 0

    for _ in range(max_new_tokens):
        # Ensure the input sequence length is max_seq_len, padding if necessary
        inp_list = cur[-max_seq_len:]
        if len(inp_list) < max_seq_len:
             pad_value = tokenizer.unk_index if hasattr(tokenizer, 'unk_index') else 0
             inp_list = [pad_value] * (max_seq_len - len(inp_list)) + inp_list

        inp = torch.tensor([inp_list], dtype=torch.long, device=DEVICE)


        token_wavelets, token_params = emitter(inp)
        semantic_params = semantic_gen(token_params)
        composite = composer(token_wavelets, token_params, semantic_params)
        pred_token_params, _ = decomposer(composite)

        last_pred_params = pred_token_params[:, -1, :, :]
        logits = classifier(last_pred_params).squeeze(0)

        # Apply temperature
        logits = logits / temperature
        # Apply Top-K and Top-P sampling
        if top_k is not None:
             v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
             logits[logits < v[-1]] = -float('inf')
        if top_p is not None:
             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
             sorted_indices_to_remove = cumulative_probs > top_p
             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
             sorted_indices_to_remove[..., 0] = 0
             indices_to_remove = sorted_indices[sorted_indices_to_remove]
             logits[indices_to_remove] = -float('inf')

        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()


        if next_token == last_token:
            repeat_count += 1
            if repeat_count >= 3:
                break
        else:
            repeat_count = 0

        last_token = next_token
        cur.append(next_token)

        stop_token_index = tokenizer.unk_index if not hasattr(tokenizer, 'pad_index') or tokenizer.pad_index == tokenizer.unk_index else tokenizer.unk_index
        if next_token == stop_token_index:
             # If the stop token is generated, stop generating further.
             break


    return tokenizer.decode(cur[len(seed_idxs):])