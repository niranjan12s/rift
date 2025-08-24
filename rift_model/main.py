token_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.unk_index) # Use unk_index
wave_recon_loss_fn = nn.MSELoss()
semantic_align_loss_fn = nn.MSELoss()
STEPS_PER_EPOCH = NUM_SAMPLES // BATCH_SIZE

opt = AdamW(list(emitter.parameters()) +
            list(semantic_gen.parameters()) +
            list(composer.parameters()) +
            list(decomposer.parameters()) +
            list(classifier.parameters()),
            lr=LR, weight_decay=WEIGHT_DECAY)
sched = StepLR(opt, step_size=5, gamma=0.8)

# --- Training loop ---
from tqdm import tqdm # Import tqdm

print("Starting training loop...")
global_step = 0
best_loss = float("inf")
for epoch in range(1, EPOCHS + 1):
    # Set only necessary modules to train mode
    emitter.train(); semantic_gen.train(); composer.train(); decomposer.train(); classifier.train();
    if fft_attn is not None:
        fft_attn.train() # Include fft_attn if it's not None

    epoch_loss = 0.0
    start_time = time.time()
    for i, (inp_tokens, target_next) in enumerate(tqdm(train_loader, total=STEPS_PER_EPOCH, desc=f"Epoch {epoch}/{EPOCHS}")):


        inp_tokens = inp_tokens.to(DEVICE)            # (B, S)
        target_next = target_next.to(DEVICE)          # (B,)

        # Use AMP if available
        context = nullcontext()
        with context:
            # Step 1: emitter -> wavelets
            token_wavelets, token_params = emitter(inp_tokens)          # (B, S, T, D), (B, S, D, P)
            B, S, T, D = token_wavelets.shape

            # Step 2: semantic generator -> semantic params
            semantic_params = semantic_gen(token_params)  # (B, D, P)

            # Step 3: compose full sentence (fft_attn is None)
            composite = composer(token_wavelets, token_params, semantic_params)         # (B, S, T, D)

            # Step 4: Decompose composite wave -> predict token params and semantic params
            pred_token_params, pred_semantic_params = decomposer(composite)

            # Step 5: classify the last token's *predicted* parameters to predict next token
            last_pred_params = pred_token_params[:, -1, :, :] # (B, D, P)
            logits = classifier(last_pred_params)                     # (B, vocab)


            # Cross-entropy token loss (predict next token)
            token_loss = token_loss_fn(logits, target_next)

            # Calculate other losses
            wave_recon = wave_recon_loss_fn(composite, token_wavelets.detach())
            sem_align = semantic_align_loss_fn(pred_semantic_params, semantic_params)
            diversity_penalty = torch.tensor(0.0, device=DEVICE)
            if USE_DIVERSITY_PENALTY:
                probs = F.softmax(logits, dim=-1)
                top1 = torch.argmax(probs, dim=-1)
                diversity_penalty = (top1.float().std()).neg() * 1e-3

            # Combine all losses
            loss = (LAMBDA_TOKEN * token_loss) + (LAMBDA_WAVE_RECON * wave_recon) + (LAMBDA_SEM_ALIGN * sem_align) + (LAMBDA_DIVERSITY * diversity_penalty)


        opt.zero_grad()
        loss.backward()
        # Only clip gradients for the parameters in the current optimizer
        torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']], GRAD_CLIP)
        opt.step()


        epoch_loss += loss.item()
        global_step += 1

        # if i % 100 == 0: # Removed printing every 100 steps
        #      print(f"[Epoch {epoch}/{EPOCHS} Step {i}/{STEPS_PER_EPOCH}] Loss: {loss.item():.4f}")

    # Print average loss at the end of each epoch
    avg_epoch_loss = epoch_loss / max(1, STEPS_PER_EPOCH)
    print(f"Epoch {epoch}/{EPOCHS} finished. avg_loss={avg_epoch_loss:.4f}. Time={(time.time()-start_time):.1f}s")
    sched.step()

    ckpt_state = {
        "epoch": epoch,
        "emitter": emitter.state_dict(),
        "semantic_gen": semantic_gen.state_dict(),
        "composer": composer.state_dict(),
        "decomposer": decomposer.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": opt.state_dict(),
        # Save the tokenizer's word_to_id and id_to_word
        "tokenizer": {"word_to_id": tokenizer.word_to_id, "id_to_word": tokenizer.id_to_word, "unk_token": tokenizer.unk_token, "unk_index": tokenizer.unk_index},
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "num_dimensions": NUM_DIMENSIONS,
            "wavelet_params": WAVELET_PARAMS,
            "time_steps": TIME_STEPS,
            "num_heads": 4, # Keep config for full model
            "head_dim": 64, # Keep config for full model
            "max_seq_len": MAX_SEQ_LEN,
            "hidden_dim": HIDDEN_DIM,
            "use_diversity_penalty": USE_DIVERSITY_PENALTY,
            "lambda_token": LAMBDA_TOKEN,
            "lambda_wave_recon": LAMBDA_WAVE_RECON,
            "lambda_sem_align": LAMBDA_SEM_ALIGN,
            "lambda_diversity": LAMBDA_DIVERSITY,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "num_samples": NUM_SAMPLES,
            "sample_max_tokens": SAMPLE_MAX_TOKENS,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN # Ensure unk_token is also in config
        }
    }
    # Optionally save fft_attn state if instantiated, even if not currently trained
    if fft_attn is not None:
        ckpt_state["fft_attn"] = fft_attn.state_dict()

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"specialized_token_only_epoch{epoch}.pt")
    torch.save(ckpt_state, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    # track best (based on total loss)
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        best_ckpt_state = {
            "epoch": epoch,
            "emitter": emitter.state_dict(),
            "semantic_gen": semantic_gen.state_dict(),
            "composer": composer.state_dict(),
            "decomposer": decomposer.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": opt.state_dict(),
            # Save the tokenizer's word_to_id and id_to_word
            "tokenizer": {"word_to_id": tokenizer.word_to_id, "id_to_word": tokenizer.id_to_word, "unk_token": tokenizer.unk_token, "unk_index": tokenizer.unk_index},
             "config": ckpt_state["config"] # Save the full config with the best checkpoint
        }
        if fft_attn is not None:
             best_ckpt_state["fft_attn"] = fft_attn.state_dict()

        torch.save(best_ckpt_state, os.path.join(CHECKPOINT_DIR, "specialized_token_only_best.pt"))
        print("Saved best checkpoint.")


print("Training finished.")