import nltk

nltk.download('brown')
from nltk.corpus import brown
from collections import Counter
import matplotlib.pyplot as plt

# Count the frequency of each word in the corpus
corpus_freq = Counter(brown.words())
descending = sorted(corpus_freq.items(), key=lambda x: x[1], reverse=True)
print(descending[0])
print(brown.categories())
# Choose genres
genres = ['adventure', 'humor']


# Function to count words by genre and sort them
def count_genre(genre):
    words = brown.words(categories=genre)
    genre_freq = Counter(words)
    sorted_words = sorted(genre_freq.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


for genre in genres:
    genre_list = count_genre(genre)
    print(f'The sorted list for {genre} is:', genre_list[0:10])

# Count the number of words (excluding punctuation)
words = [word for word in brown.words() if word.isalpha()]
num_words = len(words)

# Compute the average number of words per sentence
sentences = brown.sents()
num_sentences = len(sentences)
avg_words_per_sentence = num_words / num_sentences

# Compute the average word length
total_word_length = sum(len(word) for word in words)
avg_word_length = total_word_length / num_words

# POS tagging and ten most frequent POS tags
pos_tags = nltk.pos_tag(words)
pos_freq = Counter(tag for word, tag in pos_tags)
ten_most_common_pos = pos_freq.most_common(10)

# Print the results
print("Number of tokens:", len(brown.words()))
print("Number of types:", len(set(brown.words())))
print(f"Number of words: {num_words}")
print(f"Average number of words per sentence: {avg_words_per_sentence:.2f}")
print(f"Average word length: {avg_word_length:.2f}")
print("Ten most frequent POS tags:")
for tag, freq in ten_most_common_pos:
    print(f"{tag}: {freq}")


def sublot_frequency(descending_list, title):
    # Extract frequencies
    frequencies = [freq for word, freq in descending_list]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    # Linear plot
    ax1.plot(frequencies)
    ax1.set_xlabel('Position in the frequency list')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Linear Axes')

    # Log-log plot
    ax2.loglog(frequencies)
    ax2.set_xlabel('Position in the frequency list')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Log-Log Axes')

    plt.show()


sublot_frequency(descending, 'Whole Corpus')
for genre in genres:
    genre_freq = count_genre(genre)
    sublot_frequency(genre_freq, f"Genre:{genre.capitalize()}")
