import numpy as np
import codecs
from sklearn.preprocessing import normalize

vocab = codecs.open("brown_vocab_100.txt")

word_index_dict = {k.rstrip(): v for v, k in enumerate(vocab.read().splitlines())}
counts = np.zeros((len(word_index_dict), len(word_index_dict), len(word_index_dict)))# + 0.1

f = codecs.open("brown_100.txt")
for l in f.read().splitlines():
    indices = np.apply_along_axis(lambda x: np.array([word_index_dict[i.lower()] for i in x]), 1, 
                                  np.lib.stride_tricks.sliding_window_view(np.array(l.split()), 3))
    # print(indices)
    np.add.at(counts, (indices[:, 0], indices[:, 1], indices[:, 2]), 1)

unsmoothed_probs = counts / counts.sum(axis=2)[:,:, np.newaxis]
smoothed_probs = (counts + 0.1) / (counts + 0.1).sum(axis=2)[:,:, np.newaxis]
# probs = probs.reshape(len(word_index_dict), len(word_index_dict), len(word_index_dict))

spec_ind = np.array([
    [word_index_dict["in"], word_index_dict["the"], word_index_dict["past"]],
    [word_index_dict["in"], word_index_dict["the"], word_index_dict["time"]],
    [word_index_dict["the"], word_index_dict["jury"], word_index_dict["recommended"]],
    [word_index_dict["jury"], word_index_dict["said"], word_index_dict["that"],],
    [word_index_dict["agriculture"], word_index_dict["teacher"], word_index_dict[","]],
])

spec_probs_uns = unsmoothed_probs[spec_ind[:, 0], spec_ind[:, 1], spec_ind[:, 2]]
spec_probs_smo = smoothed_probs[spec_ind[:, 0], spec_ind[:, 1], spec_ind[:, 2]]
print(spec_probs_uns)
print(spec_probs_smo)



f.close()
