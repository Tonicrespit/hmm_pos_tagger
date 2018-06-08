import pandas as pd

import sklearn.metrics
from sklearn.preprocessing import MultiLabelBinarizer

from .Utils import get_tags


def confusion_matrix(true_tags, pred_tags, as_df=False):
    if as_df:
        tags = set(get_tags(true_tags, flatten=True)).union(set(get_tags(pred_tags, flatten=True)))
        tags = tuple(tags)

    true_tags, pred_tags = _pre_process(true_tags, pred_tags)

    cm = sklearn.metrics.confusion_matrix(true_tags, pred_tags)
    if not as_df:
        return cm
    else:
        cm_df = pd.DataFrame(cm)
        cm_df.columns = tags
        cm_df['Index'] = tags
        cm_df.set_index('Index', inplace=True)
        return cm_df


def accuracy(true_tags, pred_tags):
    true_tags, pred_tags = _pre_process(true_tags, pred_tags)

    return sklearn.metrics.accuracy_score(true_tags, pred_tags)


def precision(true_tags, pred_tags):
    true_tags, pred_tags = _pre_process(true_tags, pred_tags)

    return sklearn.metrics.precision_score(true_tags, pred_tags)


def recall(true_tags, pred_tags):
    true_tags, pred_tags = _pre_process(true_tags, pred_tags)

    return sklearn.metrics.recall_score(true_tags, pred_tags)


def f_score(true_tags, pred_tags):
    true_tags, pred_tags = _pre_process(true_tags, pred_tags)

    return sklearn.metrics.f1_score(true_tags, pred_tags)


def _pre_process(true_tags, pred_tags):
    true_tags = get_tags(true_tags, flatten=True)
    pred_tags = get_tags(pred_tags, flatten=True)

    if len(true_tags) != len(pred_tags):
        raise ValueError('Arrays must be of the same length.')

    tags = set(true_tags).union(set(pred_tags))
    tags = tuple(tags)

    for i in range(0, len(true_tags)):
        true_tags[i] = tags.index(true_tags[i])
        pred_tags[i] = tags.index(pred_tags[i])

    return true_tags, pred_tags
