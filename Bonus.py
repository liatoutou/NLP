import numpy as np
import codecs
from nltk.corpus import brown
from tqdm import tqdm

cleaned_words = [i.lower() for i in brown.words() if i.isalpha()]
cleaned_sentences = [[i.lower() for i in s if i.isalpha()] for s in brown.sents() if len(s) > 1]

unique_words = list(set(cleaned_words))
word_index_dict = {k.lower(): v for v, k in enumerate(unique_words)}
# counts_2 = np.zeros((len(word_index_dict), len(word_index_dict)))
counts_1 = np.zeros((len(word_index_dict)))

# f = codecs.open("brown_100.txt")
for l in cleaned_sentences:
    if len(l) == 0:
        continue
    indices_1 = np.apply_along_axis(lambda x: np.array([word_index_dict[i.lower()] for i in x]), 0, np.array(l))
    np.add.at(counts_1, indices_1, 1)

less_10 = np.where(counts_1 < 10)[0]

words_more_10 = np.array(list(word_index_dict.keys()))[~np.isin(np.arange(len(word_index_dict)), less_10)]
counts_1_removed = counts_1[~np.isin(np.arange(len(counts_1)), less_10)]

new_index_dict = {k: v for v, k in enumerate(words_more_10)}
counts_2 = np.zeros((len(counts_1_removed), len(counts_1_removed)))

N = len(new_index_dict)

# old_to_new = {word_index_dict[k]: new_index_dict[k] for k in new_index_dict.keys()}


for l in tqdm(cleaned_sentences):
    if len(l) < 2:
        continue
    indices_2 = np.array([r for r in np.lib.stride_tricks.sliding_window_view(l, 2) if r[0] in words_more_10 and r[1] in words_more_10])
    if len(indices_2) == 0:
        continue
    indices_2 = np.apply_along_axis(lambda x: np.array([new_index_dict[i.lower()] for i in x]), 1, indices_2)

    # filtered_indices_2 = indices_2[~np.any(np.isin(indices_2, less_10), axis=1)]
    np.add.at(counts_2, (indices_2[:, 0], indices_2[:, 1]), 1)

pmi = np.log((counts_2 + 0.1) * N / ((counts_1_removed[None, :] + 0.1) * (counts_1_removed[:, None] + 0.1)))

sorted_indices = np.argsort(pmi.flatten())[::-1]
top_20_indices = np.unravel_index(sorted_indices[:20], pmi.shape)
bottom_20_indices = np.unravel_index(sorted_indices[20:], pmi.shape)

inverse_word_index_dict = {v: k for k, v in new_index_dict.items()}

top_20_pairs = np.array([[inverse_word_index_dict[i], inverse_word_index_dict[j]] for i, j in zip(*top_20_indices)])
bottom_20_pairs = np.array([[inverse_word_index_dict[i], inverse_word_index_dict[j]] for i, j in zip(*bottom_20_indices)])

print(top_20_pairs)
print(bottom_20_pairs)
