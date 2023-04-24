#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


vocab = open("brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict), dtype=int)

for sentence in f:
    words = sentence.lower().split()
    for word in words:
        if word in word_index_dict:
            counts[word_index_dict[word]] += 1
f.close()

#print(counts)
probs = counts / np.sum(counts)
#print(probs)
np.savetxt('unigram_probs.txt', probs, fmt='%.8f')
toy_corpus = open("toy_corpus.txt")
output_file = open("unigram_eval.txt", "w")

for sentence in toy_corpus:
    words = sentence.lower().strip().split()
    sent_len = len(words)
    sent_prob = 1

    for word in words:
        if word in word_index_dict:
            word_prob = probs[word_index_dict[word]]
            sent_prob *= word_prob
    print(sent_prob)

    perplexity = 1 / pow(sent_prob, 1.0 / sent_len)

    output_file.write(f"{perplexity}\n")

output_file.close()
toy_corpus.close()
