from collections import defaultdict

import numpy as np

from util import accuracy
from hmm import HMM


def normalization(vector):
    norm = sum(vector)
    return vector / norm


# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    # Edit here
    low = 10 ** (-6)
    states = defaultdict(int)
    for i, tag in enumerate(tags):
        states[tag] = i
    ids = 1
    wordVsId = defaultdict(int)
    for sentences in train_data:
        for word in sentences.words:
            if word not in wordVsId:
                wordVsId[word] = ids
                ids = ids + 1

    S = len(states)
    L = len(wordVsId) + 1
    pi = np.zeros(S)
    A = np.zeros([S, S])
    B = np.zeros([S, L])

    for data in train_data:
        pi[states[data.tags[0]]] = pi[states[data.tags[0]]] + 1
        for state, state_next in zip(data.tags, data.tags[1:]):
            A[states[state], states[state_next]] = A[states[state], states[state_next]] + 1
        for state, observation in zip(data.tags, data.words):
            B[states[state], wordVsId[observation]] = B[states[state], wordVsId[observation]] + 1

    zeros_A = np.zeros_like(A)
    A = np.divide(A, np.sum(A, axis=1), out=zeros_A, where=np.sum(A, axis=1) != 0)

    zeros_B = np.zeros_like(B)
    B = np.divide(B, np.sum(B, axis=1)[:, np.newaxis], out=zeros_B, where=np.sum(B, axis=1)[:, np.newaxis] != 0)

    B[:, 0] = low * np.ones(S)

    pi = normalization(pi)

    model = HMM(pi, A, B, wordVsId, states)
    ###################################################
    return model


# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    for data in test_data:
        word_data = data.words
        result = model.viterbi(word_data)
        tagging.append(result)
    ###################################################
    return tagging
