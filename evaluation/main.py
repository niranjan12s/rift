def calculate_bleu(reference, hypothesis):
    """Calculates BLEU score. Reference and hypothesis are tokenized lists of strings."""
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], hypothesis, smoothing_function=smoothie)

def calculate_repetition_score(tokens):
    """Calculates a simple repetition score based on token frequency."""
    if not tokens:
        return 0.0
    token_counts = Counter(tokens)
    # Score is the number of tokens that appear more than once, normalized by total tokens
    repeated_tokens = sum(count for token, count in token_counts.items() if count > 1)
    return repeated_tokens / len(tokens)

def calculate_ngram_diversity(tokens, n):
    """Calculates n-gram diversity (unique n-grams / total n-grams)."""
    if len(tokens) < n:
        return 0.0
    ngrams = list(nltk.ngrams(tokens, n))
    if not ngrams:
        return 0.0
    unique_ngrams = set(ngrams)
    return len(unique_ngrams) / len(ngrams)

def clean_generated_text(text):
    """Basic cleaning for evaluation."""
    # Remove special tokens
    text = text.replace(PAD_TOKEN, "").replace(UNK_TOKEN, "")
    # Remove leading/trailing whitespace
    text = text.strip()
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.lower() # Convert to lowercase for case-insensitive comparison


# --- Static Prompts for Evaluation ---
# Create a list of 20-30 prompts based on potential training data themes
evaluation_prompts = [
    "The quick brown fox jumps over the",
    "In the realm of fantasy, magic",
    "The ancient ruins whispered stories of",
    "Artificial intelligence is rapidly transforming",
    "The history of art reflects",
    "Quantum computing promises to revolutionize",
    "Exploring the depths of the ocean reveals",
    "The future of renewable energy",
    "The human brain is a complex",
    "Understanding climate change requires",
    "The impact of technology on society",
    "Space exploration continues to push",
    "The intricate patterns of nature",
    "The challenges of modern education",
    "The beauty of a sunset paints",
    "The science of genetics has opened",
    "The philosophy of mind explores",
    "The evolution of language is a",
    "The role of literature in culture",
    "The principles of physics govern"
    # Add more prompts if needed to reach 20-30
]


# --- Generate Text and Evaluate ---

# Paths to your trained models
# Corrected paths to saved checkpoints
WAVE_CHECKPOINT = "/content/checkpoints/specialized_token_only_epoch20.pt" # Corrected path
TRANSFORMER_CHECKPOINT = "/content/transformer_checkpoints/transformer_epoch_20.pt" # Corrected path

wave_generated_texts = []
transformer_generated_texts = []

# Reduce the number of words to generate for evaluation
reduced_max_tokens = 20 # You can adjust this number


print("\n--- Generating Text for Evaluation ---")
for prompt in evaluation_prompts:
    print(f"\nPrompt: {prompt}")

    # Generate text with Wave Model
    try:
        # Use reduced_max_tokens
        gen_text_wave = generate_wave_text(prompt, checkpoint_path=WAVE_CHECKPOINT, max_new_tokens=reduced_max_tokens)
        cleaned_text_wave = clean_generated_text(gen_text_wave)
        print(f"  Wave Gen: {cleaned_text_wave}")
        wave_generated_texts.append(cleaned_text_wave)
    except FileNotFoundError:
        print(f"  Error: Wave model checkpoint not found at {WAVE_CHECKPOINT}")
        wave_generated_texts.append("") # Append empty string on error
    except Exception as e:
        print(f"  Error generating text with Wave model: {e}")
        wave_generated_texts.append("") # Append empty string on error


    # Generate text with Transformer Model
    try:
        # Use reduced_max_tokens
        gen_text_transformer = generate_transformer_text(prompt, checkpoint_path=TRANSFORMER_CHECKPOINT, max_new_tokens=reduced_max_tokens)
        cleaned_text_transformer = clean_generated_text(gen_text_transformer)
        print(f"  Transformer Gen: {cleaned_text_transformer}")
        transformer_generated_texts.append(cleaned_text_transformer)
    except FileNotFoundError:
        print(f"  Error: Transformer model checkpoint not found at {TRANSFORMER_CHECKPOINT}")
        transformer_generated_texts.append("") # Append empty string on error
    except Exception as e:
        print(f"  Error generating text with Transformer model: {e}")
        transformer_generated_texts.append("") # Append empty string on error


# --- Calculate and Display Metrics ---

print("\n--- Evaluation Results ---")

# Prepare data for metrics
# For BLEU and n-grams, we need tokenized lists
wave_tokenized = [nltk.word_tokenize(text) for text in wave_generated_texts]
transformer_tokenized = [nltk.word_tokenize(text) for text in transformer_generated_texts]
# For BERTScore, we need lists of strings
wave_text_list = wave_generated_texts
transformer_text_list = transformer_generated_texts


# Repetition Score
print("\nRepetition Score (lower is better):")
# Ensure lists are not empty before calculating mean
avg_wave_repetition = np.mean([calculate_repetition_score(tokens) for tokens in wave_tokenized]) if wave_tokenized else 0.0
avg_transformer_repetition = np.mean([calculate_repetition_score(tokens) for tokens in transformer_tokenized]) if transformer_tokenized else 0.0
print(f"Average Wave Model Repetition Score: {avg_wave_repetition:.4f}")
print(f"Average Transformer Model Repetition Score: {avg_transformer_repetition:.4f}")

# N-gram Diversity
print("\nN-gram Diversity (higher is better):")
ngram_orders = [1, 2, 3, 4] # Unigram, Bigram, Trigram, 4-gram

for n in ngram_orders:
    # Ensure lists are not empty before calculating mean
    avg_wave_ngram_diversity = np.mean([calculate_ngram_diversity(tokens, n) for tokens in wave_tokenized]) if wave_tokenized else 0.0
    avg_transformer_ngram_diversity = np.mean([calculate_ngram_diversity(tokens, n) for tokens in transformer_tokenized]) if transformer_tokenized else 0.0
    print(f"Average Wave Model {n}-gram Diversity: {avg_wave_ngram_diversity:.4f}")
    print(f"Average Transformer Model {n}-gram Diversity: {avg_transformer_ngram_diversity:.4f}")

print("\n--- Evaluation Complete ---")