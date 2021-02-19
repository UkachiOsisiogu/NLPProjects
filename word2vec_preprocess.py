from spacy.lang.en import English
import pandas as pd
import numpy as np
import pickle
from gensim.models.word2vec import Word2Vec

GLOVE_6B_50D_PATH = "files/glove_6B.txt"
GLOVE_27B_200D_PATH = "files/glove.twitter.27B.200d.txt"
ENCODING="utf-8"


def store_data(filepath, data):
    """
    This function is used for object serialization just to store what is going on
    Parameters
    ----------
    filepath: str The path where data is stored
    data: The data being stored

    Returns
    -------

    """
    pickle.dump(data, open(filepath, "wb"))
    print(f"Data stored successfully in {filepath}")


def load_file(filepath):
    """
    This function is used to load a file from the specified file path
    This was used to load the mapping dictionaries for this script
    Parameters
    ----------
    filepath: str

    Returns
    Any file
    -------

    """

    with open(filepath, 'rb') as f:
        file = pickle.load(f)
        return file


def preprocess(name, data_set):
    X = data_set.review
    y = data_set.sentiment

    nlp = English()
    token_list = []
    X_data = []

    for review in X:
        my_review = nlp(review)
        token_review = []
        for token in my_review:
            token_list.append(token.text) # All tokens
            token_review.append(token.text) # Token for each review
        X_data.append(token_review)

    # Get the vocabulary of the data_set
    all_words = set(token_list)
    print(f"The length of the vocabulary is {len(all_words)}")

    # Store the data_set and all the words
    store_data(f"files/{name}_X_data.pkl", X_data)
    store_data(f"files/{name}_y_data.pkl", y)
    store_data(f"files/{name}_allwords.pkl", all_words)

    return all_words, X_data, y  # Return the vocabulary and the X_data


def get_word_vectors(data_set_name, vocabulary, processed_data):
    # Training Word2Vec
    model = Word2Vec(processed_data, size=100, window=5, min_count=5, workers=2)
    w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

    store_data(f"files/{data_set_name}_w2v.pkl", w2v)

    # Reading GLOVE files
    glove_small = {}
    with open(GLOVE_6B_50D_PATH, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode(ENCODING)
            nums = np.array(parts[1:], dtype=np.float32)
            if word in vocabulary:
                glove_small[word] = nums

    store_data(f"files/{data_set_name}_glove_small.pkl", glove_small)

    glove_big = {}
    with open(GLOVE_27B_200D_PATH, "rb") as infile:
        for line in infile:
            parts = line.split()
            word = parts[0].decode(ENCODING)
            nums = np.array(parts[1:], dtype=np.float32)
            if word in vocabulary:
                glove_big[word] = nums

    store_data(f"files/{data_set_name}_glove_big.pkl", glove_big)

    return w2v, glove_big, glove_small


