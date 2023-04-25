import numpy as np
import codecs

vocab = codecs.open("brown_vocab_100.txt")

word_index_dict = {k.rstrip(): v for v, k in enumerate(vocab.read().splitlines())}
counts_2 = np.zeros((len(word_index_dict), len(word_index_dict)))
counts_1 = np.zeros((len(word_index_dict)))

f = codecs.open("brown_100.txt")
f_ = lambda x: np.array([word_index_dict[i.lower()] for i in x])
for l in f.read().splitlines():
    indices_1 = np.apply_along_axis(f_, 0, np.array(l.split()))
    np.add.at(counts_1, indices_1, 1)

N = len(word_index_dict)
less_10 = np.where(counts_1 < 10)[0]

f.seek(0)  # Reset the file pointer to the beginning of the file
for l in f.read().splitlines():
    word_indices = np.array(l.lower().split())
    indices_2 = np.lib.stride_tricks.sliding_window_view(word_indices, 2)
    indices_2 = np.apply_along_axis(f_, 1, indices_2)

    filtered_indices_2 = indices_2[~np.any(np.isin(indices_2, less_10), axis=1)]
    np.add.at(counts_2, (filtered_indices_2[:, 0], filtered_indices_2[:, 1]), 1)

pmi = np.log(counts_2 * N / (counts_1[None, :] * counts_1[:, None]))

top_20_indices = np.unravel_index(np.argpartition(pmi.flatten(), -20)[-20:], pmi.shape)
bottom_20_indices = np.unravel_index(np.argpartition(pmi.flatten(), 20)[:20], pmi.shape)

inverse_word_index_dict = {v: k for k, v in word_index_dict.items()}

top_10_pairs = np.array([[inverse_word_index_dict[i], inverse_word_index_dict[j]] for i, j in zip(*top_20_indices)])
bottom_10_pairs = np.array([[inverse_word_index_dict[i], inverse_word_index_dict[j]] for i, j in zip(*bottom_20_indices)])

print(top_10_pairs)
print(bottom_10_pairs)
