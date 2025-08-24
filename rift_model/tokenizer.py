import re
from collections import Counter

class WordLevelTokenizer:
    def __init__(self, dataset_size=None, max_sentence_length=None, min_word_frequency=1, unk_token='<unk>', max_vocab_size=None):
        self.dataset_size = dataset_size
        self.max_sentence_length = max_sentence_length
        self.min_word_frequency = min_word_frequency
        self.unk_token = unk_token
        self.max_vocab_size = max_vocab_size # New parameter
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.unk_index = 0

    def build_vocabulary(self, texts):
        if self.dataset_size is not None:
            texts = texts[:self.dataset_size]

        word_counts = Counter()
        for text in texts:
            words = self._preprocess_text(text)
            word_counts.update(words)

        # Filter words based on minimum frequency
        filtered_words = [word for word, count in word_counts.items() if count >= self.min_word_frequency]

        # Sort words by frequency in descending order and take top N if max_vocab_size is set
        if self.max_vocab_size is not None:
            # Sort by count (descending), then by word (ascending) for tie-breaking
            sorted_word_counts = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
            # Filter sorted words based on minimum frequency
            filtered_and_sorted_words = [word for word, count in sorted_word_counts if count >= self.min_word_frequency]
            # Take the top N words, leaving space for the UNK token
            top_words = filtered_and_sorted_words[:self.max_vocab_size - 1]
        else:
            top_words = filtered_words

        # Add the unknown token first
        self.word_to_id[self.unk_token] = 0
        self.id_to_word[0] = self.unk_token
        self.vocab_size = 1

        # Build vocabulary from top words, starting from id 1
        for word in top_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = self.vocab_size
                self.id_to_word[self.vocab_size] = word
                self.vocab_size += 1

        # Ensure vocab_size reflects the actual number of unique tokens including the unk token
        self.vocab_size = len(self.word_to_id)
        print(f"Built vocabulary with size: {self.vocab_size}")


    def tokenize(self, text):
        words = self._preprocess_text(text)
        if self.max_sentence_length is not None:
            words = words[:self.max_sentence_length]
        token_ids = [self.word_to_id.get(word, self.unk_index) for word in words]
        return token_ids

    def _preprocess_text(self, text):
        text = text.lower()
        # Remove characters that are NOT lowercase letters, numbers, or whitespace
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        return words

    def decode(self, token_ids):
        words = [self.id_to_word.get(token_id, self.unk_token) for token_id in token_ids]
        return " ".join(words)