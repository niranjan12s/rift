@torch.no_grad()
def generate_transformer_text(prompt: str, checkpoint_path: str, max_new_tokens=SAMPLE_MAX_TOKENS, temp=1.0):
    print(f"Loading Transformer checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = GPTLanguageModel().to(DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    # Get sampling parameters from config, with defaults
    config = {} # Assuming config is not saved with Transformer checkpoint, use defaults or get from Wave config if available
    # For consistency, try to get config from Wave checkpoint if possible, otherwise use defaults
    try:
        wave_ckpt_path = "/content/checkpoints/specialized_token_only_best.pt" # Or a specific epoch checkpoint
        wave_checkpoint = torch.load(wave_ckpt_path, map_location=DEVICE)
        config = wave_checkpoint.get("config", {})
    except:
        print("Wave model config not found, using default sampling parameters.")
        pass # Use empty config if loading fails

    top_k = config.get("top_k", 50)
    top_p = config.get("top_p", 0.9)
    temperature = config.get("temperature", 1.0)
    max_seq_len = config.get("max_seq_len", MAX_SEQ_LEN) # Get max_seq_len from config


    # Use tokenizer.tokenize instead of encode
    seed_idxs = tokenizer.tokenize(prompt)
     # Truncate seed_idxs to max_seq_len
    seed_idxs = seed_idxs[:max_seq_len] # Use max_seq_len from config


    cur = torch.tensor([seed_idxs], dtype=torch.long, device=DEVICE)
    last_token = -1 # Initialize last_token to a value that won't match a token ID
    repeat_count = 0

    # Implement generation loop directly
    for i in range(max_new_tokens): # Use i for iteration count
        # Ensure the input sequence length is max_seq_len, padding if necessary
        inp_list = cur.squeeze(0).tolist()
        inp_list = inp_list[-max_seq_len:] # Use max_seq_len

        if len(inp_list) < max_seq_len: # Use max_seq_len
             pad_value = tokenizer.unk_index if hasattr(tokenizer, 'unk_index') else 0 # Use unk_index for padding if pad_index is not available/different
             inp_list = [pad_value] * (max_seq_len - len(inp_list)) + inp_list # Use max_seq_len

        idx_cond = torch.tensor([inp_list], dtype=torch.long, device=DEVICE)

        # Get predictions from the model's forward pass
        logits, _ = model(idx_cond) # Pass the input sequence to the model

        # Focus only on the last time step (the prediction for the next token)
        logits = logits[:, -1, :] # Shape (B, vocab_size), where B is 1

        # Apply temperature
        logits = logits / temperature
        # Apply Top-K and Top-P sampling
        if top_k is not None:
             # Correct indexing for top_k filtering threshold
             v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # logits is (1, 200), v is (1, top_k)
             logits[logits < v[0, -1]] = -float('inf') # Corrected from v[-1] to v[0, -1]

        if top_p is not None:
             sorted_logits, sorted_indices = torch.sort(logits, descending=True)
             cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
             sorted_indices_to_remove = cumulative_probs > top_p
             # Shift indices to the right to keep at least one token
             sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
             sorted_indices_to_remove[..., 0] = 0
             # Apply the boolean mask directly
             logits[sorted_indices_to_remove] = -float('inf') # Corrected indexing


        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()


        # Repetition check - only perform after the first token is generated (i > 0)
        if i > 0 and next_token == last_token:
            repeat_count += 1
            if repeat_count >= 10:
                break
        else:
            repeat_count = 0

        last_token = next_token # Update last_token for the next iteration
        cur = torch.cat([cur, torch.tensor([[next_token]], device=DEVICE)], dim=1)

        # Remove the stop condition check based on next_token
        # stop_token_index = tokenizer.unk_index if not hasattr(tokenizer, 'pad_index') or tokenizer.pad_index == tokenizer.unk_index else tokenizer.unk_index
        # if next_token == stop_token_index:
        #      # If the stop token is generated, stop generating further.
        #      break


    generated_ids = cur.squeeze(0).tolist()
    return tokenizer.decode(generated_ids[len(seed_idxs):])