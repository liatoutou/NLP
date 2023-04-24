#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

word_index_dict = {}
with open("brown_vocab_100.txt", "r") as vf:
    for idx, line in enumerate(vf):
        word = line.rstrip()
        word_index_dict[word] = idx

with open("word_to_index_100.txt", "w") as wf:
    word_to_index_str = str(word_index_dict)
    wf.write(word_to_index_str)

print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
