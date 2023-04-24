import numpy as np
from sklearn.preprocessing import normalize
import codecs

vocab = codecs.open("brown_vocab_100.txt")

# Load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i

f = codecs.open("brown_100.txt")

# Create trigram counts matrix
trigram_counts = np.zeros((len(word_index_dict), len(word_index_dict), len(word_index_dict)), dtype=float)

# Update trigram counts from the corpus
for sentence in f:
    words = sentence.lower().split()
    for i in range(len(words) - 2):
        if words[i] in word_index_dict and words[i+1] in word_index_dict and words[i+2] in word_index_dict:
            trigram_counts[word_index_dict[words[i]], word_index_dict[words[i+1]], word_index_dict[words[i+2]]] += 1

f.close()
unsmoothed_probs = trigram_counts / trigram_counts.sum()

# Add alpha for smoothing
alpha = 0.1
smoothed_counts = trigram_counts + alpha
smoothed_probs = smoothed_counts / smoothed_counts.sum()
trigrams = [
    ("in", "the", "past"),
    ("in", "the", "time"),
    ("the", "jury", "said"),
    ("the", "jury", "recommended"),
    ("jury", "said", "that"),
    ("agriculture", "teacher", ",")
]
for trigram in trigrams:
    unsmoothed_prob = unsmoothed_probs[word_index_dict[trigram[0]], word_index_dict[trigram[1]], word_index_dict[trigram[2]]]
    smoothed_prob = smoothed_probs[word_index_dict[trigram[0]], word_index_dict[trigram[1]], word_index_dict[trigram[2]]]
    print(f"Unsmoothed p({trigram[2]} | {trigram[0]}, {trigram[1]}): {unsmoothed_prob}")
    print(f"Smoothed p({trigram[2]} | {trigram[0]}, {trigram[1]}): {smoothed_prob}")
    print()