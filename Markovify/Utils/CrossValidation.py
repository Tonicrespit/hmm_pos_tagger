from .Scoring import accuracy

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


def cros_validation_score(model, sentence=None, sentences=None, root=None, fileids='.*', encoding='utf8', cv=5,
                          return_model=False, verbose=False):
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

    scores = np.zeros(cv)
    fold = 0
    for train_index, test_index in split(len(corpus), cv):
        if verbose:
            print('Training fold {} of {}.'.format(fold + 1, cv))

        train = list(corpus[i] for i in train_index)
        test = list(corpus[i] for i in test_index)
        test_words = _remove_tags(test)

        m = model.copy()
        m = m.fit(train)
        if verbose:
            print('Making predictions...')

        prediction = m.predict(test_words)
        predicted = [tag for sentence in prediction for tag in sentence]
        real = [tag for sentence in test for (word, tag) in sentence]

        model_score = accuracy(predicted, real)
        scores[fold] = model_score
        if return_model and model_score >= np.max(scores):
            best_model = m

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
    socres, best_model = cros_validation_score(model, sentence, sentences, root, fileids, encoding, cv, verbose)
    return best_model


def _remove_tags(corpus):
    """
    Removes the tags from a corpus, returning only the words.

    :param corpus: List of (lists of) words.
    :return: corpus without tags.
    """
    if type(corpus[0]) is tuple:
        words = [word for (word, tag) in corpus]
    else:
        words = []
        for sentence in corpus:
            words.append(list(word for (word, tag) in sentence))

    return words

