from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
import string


def tokenize_and_process(text, vocab_size=10000):

    text_clean = []

    stop = list(string.punctuation)

    for t in text:
        text_clean.append(" ".join([i for i in word_tokenize(t.lower()) if i not in stop and i[0] != "'"]))

    # Instantiate tokenizer
    T = Tokenizer(num_words=vocab_size)

    # Fit the tokenizer with text
    T.fit_on_texts(text_clean)

    # Turn our input text into sequences of index integers
    data = T.texts_to_sequences(text_clean)

    word_to_idx = T.word_index
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    return data, word_to_idx, idx_to_word, T

def tokenize_and_test(text, T) :
    text_clean = []
    data = []

    stop = list(string.punctuation)

    for t in text:
        text_clean.append(" ".join([i for i in word_tokenize(t.lower()) if i not in stop and i[0] != "'"]))

    data = T.texts_to_sequences(text_clean)
    return data
