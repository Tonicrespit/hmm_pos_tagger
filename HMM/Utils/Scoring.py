import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import sklearn.metrics
from sklearn.preprocessing import MultiLabelBinarizer

from .Utils import get_tags


def confusion_matrix(true_tags, pred_tags, sample_weight=None, normalize=True):
    true_tags, pred_tags, tags = _pre_process(true_tags, pred_tags, binarize=False)

    cm = sklearn.metrics.confusion_matrix(true_tags, pred_tags, labels=tags, sample_weight=sample_weight)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm = pd.DataFrame(cm, index=tags, columns=tags)
    return cm


def accuracy(true_tags, pred_tags, normalize=True, sample_weight=None):
    true_tags, pred_tags, tags = _pre_process(true_tags, pred_tags)

    return sklearn.metrics.accuracy_score(true_tags, pred_tags, normalize, sample_weight)


def precision(true_tags, pred_tags, average='weighted', sample_weight=None):
    true_tags, pred_tags, tags = _pre_process(true_tags, pred_tags, binarize=False)

    return sklearn.metrics.precision_score(true_tags, pred_tags,
                                           labels=tags, average=average, sample_weight=sample_weight)


def recall(true_tags, pred_tags, average='weighted', sample_weight=None):
    true_tags, pred_tags, tags = _pre_process(true_tags, pred_tags, binarize=False)

    return sklearn.metrics.recall_score(true_tags, pred_tags,
                                        labels=tags, average=average, sample_weight=sample_weight)


def f_score(true_tags, pred_tags, average='weighted', sample_weight=None):
    true_tags, pred_tags, tags = _pre_process(true_tags, pred_tags, binarize=False)

    return sklearn.metrics.f1_score(true_tags, pred_tags,
                                    labels=tags, average=average, sample_weight=sample_weight)


def _pre_process(true_tags, pred_tags, binarize=True):
    true_tags = get_tags(true_tags, flatten=True)
    pred_tags = get_tags(pred_tags, flatten=True)

    if len(true_tags) != len(pred_tags):
        raise ValueError('Arrays must be of the same length.')

    tags = set(true_tags).union(set(pred_tags))
    tags = tuple(tags)

    if binarize:
        for i in range(0, len(true_tags)):
            true_tags[i] = tags.index(true_tags[i])
            pred_tags[i] = tags.index(pred_tags[i])

    return true_tags, pred_tags, tags

