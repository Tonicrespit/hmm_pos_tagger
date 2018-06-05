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


def cross_val_score(model, sentence=None, sentences=None, root=None, fileids='.*', encoding='utf8', cv=3, verbose=False):
    """
    Performs a cross-validation training to the model and returns the best accuracy.

    :param model: markovify object.
    :param sentence: List of tuples (word, tag).
    :param sentences: List of sentences.
    :param root: Folder with the text files indise.
    :param fileids: Files to read in the root. By default, '.*' which means 'all'.
    :param encoding: Encoding of the files.
    :param cv: Number of folds for the cross-validation.
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

    accuracies = []
    fold = 1
    for train_index, test_index in split(len(corpus), cv):
        if verbose:
            print('Fold {} of {}.'.format(fold, cv))
        train = list(corpus[i] for i in train_index)
        test = list(corpus[i] for i in test_index)
        test_words = _remove_tags(test)

        trained = model.fit(train)
        prediction = trained.predict(test_words)
        predicted = [tag for sentence in prediction for tag in sentence]
        actual = [tag for sentence in test for (word, tag) in sentence]
        accuracies.append(_accuracy(predicted, actual))

        if verbose:
            print('Ending fold {} of {}. Accuracy of the fold: {}.'.format(fold, cv, accuracies[fold - 1]))
        fold += 1

    return accuracies


def _accuracy(predicted, actual):
    if len(predicted) != len(actual):
        raise ValueError('Arrays must be of the same length.')

    total_words = len(predicted)
    correct = 0
    for i in range(0, total_words):
        if predicted[i] == actual[i]:
            correct += 1
    return correct / total_words


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

