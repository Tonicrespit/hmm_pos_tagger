from .Scoring import *
from .Utils import get_words

import numpy as np
from sklearn.model_selection import KFold


def split(nsentences, cv):
    """

    :param nsentences:
    :param cv:
    :return:
    """
    index = np.arange(nsentences)
    for i in range(0, cv):
        lower = np.floor((i/cv) * nsentences)
        upper = np.floor(((i + 1)/cv) * nsentences)
        test_range = np.arange(lower, upper, dtype=np.int16)

        mask_test = np.zeros(nsentences, dtype=np.bool)
        mask_test[test_range] = True
        mask_train = np.logical_not(mask_test)

        test_index = index[mask_test]
        train_index = index[mask_train]
        yield train_index, test_index


def cross_validation_score(model, sentence=None, sentences=None, root=None, fileids='.*', encoding='utf8', cv=5,
                           return_model=False, verbose=False, n_jobs=8):
    """
        Performs a cross-validation training to the model and returns the best accuracy.

        :param model: Markovify object.
        :param sentence: List of tuples (word, tag).
        :param sentences: List of sentences.
        :param root: Folder with the text files indise.
        :param fileids: Files to read in the root. By default, '.*' which means 'all'.
        :param encoding: Encoding of the files.
        :param cv: Number of folds for the cross-validation.
        :param return_model: True if you want to get the best model. False otherwise.
        :param verbose: Get additional information of the training via standard output.
        :param n_jobs: Number of CPUs used to compute the predictions.
        :return: List of accuracies.
        """
    if sentence is not None:
        corpus = sentence
    elif sentences is not None:
        corpus = sentences
    elif root is not None:
        reader = CorpusParser(root, fileids, encoding)
        corpus = reader.sentences()
    else:
        raise ValueError('Sentence, sentences or root must be provided.')

    if len(corpus) < cv:
        raise ValueError('A corpus of lesser length than folds cannot be cross-validated.')

    scores = []
    max_score = -1
    fold = 0
    for train_index, test_index in split(len(corpus), cv):
        if verbose:
            print('Training fold {} of {}.'.format(fold + 1, cv))

        train = list(corpus[i] for i in train_index)
        test = list(corpus[i] for i in test_index)

        m = model.copy()
        m = m.fit(train)
        if verbose:
            print('Making predictions...')

        predicted_taggs = m.predict(test, n_jobs=n_jobs)

        model_score = accuracy(test, predicted_taggs)
        scores.append({
            'accuracy': accuracy(test, predicted_taggs),
            'precision': precision(test, predicted_taggs),
            'recall': recall(test, predicted_taggs)
        })

        if return_model and model_score >= max_score:
            best_model = m
            max_score = model_score

        if verbose:
            print('Ending fold {} of {}. Score: {}.'.format(fold + 1, cv, scores[fold]))

        fold += 1
    if return_model:
        return scores, best_model
    else:
        return scores


def cross_validation(model, sentence=None, sentences=None, root=None, fileids='.*', encoding='utf8',
                     cv=5, verbose=False):
    """
    Performs a cross-validation training to the model and returns the best.

    :param model: Markovify object.
    :param sentence: List of tuples (word, tag).
    :param sentences: List of sentences.
    :param root: Folder with the text files indise.
    :param fileids: Files to read in the root. By default, '.*' which means 'all'.
    :param encoding: Encoding of the files.
    :param cv: Number of folds for the cross-validation.
    :param verbose: Get additional information of the training via standard output.
    :return: List of accuracies.
    """
    socres, best_model = cross_validation_score(model, sentence, sentences, root, fileids, encoding, cv, verbose)
    return best_model
