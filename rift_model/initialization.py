fft_attn = MultiHeadFFTAttention(time_steps=TIME_STEPS, num_dimensions=NUM_DIMENSIONS, num_heads=4, head_dim=64).to(DEVICE)

emitter = WaveletGenerator(tokenizer.vocab_size, num_dimensions=NUM_DIMENSIONS, wavelet_params=WAVELET_PARAMS, time_steps=TIME_STEPS).to(DEVICE)
semantic_gen = SemanticWaveGenerator(num_dimensions=NUM_DIMENSIONS, wavelet_params=WAVELET_PARAMS, hidden_dim=HIDDEN_DIM).to(DEVICE)
composer = WaveSentenceComposer(time_steps=TIME_STEPS, num_dimensions=NUM_DIMENSIONS, fft_attn=fft_attn).to(DEVICE)
decomposer = WaveDecomposer(num_tokens=MAX_SEQ_LEN, num_dimensions=NUM_DIMENSIONS, wavelet_params=WAVELET_PARAMS, hidden_dim=HIDDEN_DIM).to(DEVICE)
classifier = WaveTokenClassifier(num_dimensions=NUM_DIMENSIONS, wavelet_params=WAVELET_PARAMS, vocab_size=tokenizer.vocab_size, hidden_dim=HIDDEN_DIM).to(DEVICE)
# print model sizes for diagnostics
def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Emitter params: {count_params(emitter):,}")
print(f"Semantic params: {count_params(semantic_gen):,}")
print(f"Composer params: {count_params(composer):,}")
print(f"Decomposer params: {count_params(decomposer):,}")
print(f"Classifier params: {count_params(classifier):,}")
print(f"FFT Attn params: {count_params(fft_attn):,}")
