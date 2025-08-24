#Handling data
print("Loading training file and building tokenizer...")
raw_lines = []
with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    # Read lines until we have exactly 100 non-empty ones for vocabulary building AND training data
    for line in f:
        stripped_line = line.strip()
        if stripped_line:
            raw_lines.append(stripped_line)
        if len(raw_lines) >= 100: # Limit to the first 100 sentences for vocabulary building and training
            break

# Initialize tokenizer with the specified dataset size and max vocabulary size
# Use all 100 lines for vocabulary building
tokenizer = WordLevelTokenizer(dataset_size=100, max_vocab_size=200)
tokenizer.build_vocabulary(raw_lines) # Build vocabulary using the first 100 sentences

# Use the first 100 lines as the training data for the dataset
# NUM_SAMPLES is now read from the config (cell FMgakxBejvFZ) and is 8000.
# We no longer dynamically calculate it here.
# The TextWaveDataset will sample NUM_SAMPLES windows randomly from the raw_lines.


train_dataset = TextWaveDataset(lines=raw_lines, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN, num_samples=NUM_SAMPLES)

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True,
                          num_workers=2, # Use num_workers as suggested
                          pin_memory=True if torch.cuda.is_available() else False) # Use pin_memory

# Print confirmation of NUM_SAMPLES being used
print(f"Using NUM_SAMPLES from config: {NUM_SAMPLES:,}")