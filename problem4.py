#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
import codecs

vocab = codecs.open("brown_vocab_100.txt")

# load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    word = line.rstrip()
    word_index_dict[word] = i

f = codecs.open("brown_100.txt")

counts = counts = np.zeros((len(word_index_dict), len(word_index_dict)), dtype=float)
for sentence in f:
    words = sentence.lower().split()
    previous_word = '<s>'
    for word in words[1:]:
        if word in word_index_dict and previous_word in word_index_dict:
            counts[word_index_dict[previous_word], word_index_dict[word]] += 1
        previous_word = word
counts += 0.1

smooth_probs = normalize(counts, norm='l1', axis=1)

with open("smooth_probs.txt", "w") as bp:
    bp.write(f"p(the | all): {smooth_probs[word_index_dict['all'], word_index_dict['the']]:.7f}\n")
    bp.write(f"p(jury | the): {smooth_probs[word_index_dict['the'], word_index_dict['jury']]:.7f}\n")
    bp.write(f"p(campaign | the): {smooth_probs[word_index_dict['the'], word_index_dict['campaign']]:.7f}\n")
    bp.write(f"p(calls | anonymous): {smooth_probs[word_index_dict['anonymous'], word_index_dict['calls']]:.7f}\n")

f.close()
toy_corpus = open("toy_corpus.txt")
output_file = open("smoothed_eval.txt", "w")

for sentence in toy_corpus:
    words = sentence.lower().strip().split()
    sent_len = len(words) - 1
    sent_prob = 1
    previous_word = '<s>'

    for word in words[1:]:
        if word in word_index_dict and previous_word in word_index_dict:
            word_prob = smooth_probs[word_index_dict[previous_word], word_index_dict[word]]
            sent_prob *= word_prob
        previous_word = word

    perplexity = 1 / pow(sent_prob, 1.0 / sent_len)

    output_file.write(f"{perplexity}\n")

output_file.close()
toy_corpus.close()
generated_sentences = open("smoothed_generation.txt", "w")
for _ in range(10):
    sentence = GENERATE(word_index_dict, smooth_probs, "bigram", 50, "<s>")
    generated_sentences.write(sentence.strip() + "\n")
generated_sentences.close()
