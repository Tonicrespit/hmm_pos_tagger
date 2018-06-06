from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def confusion_matrix(true_class, pred_class):
    return confusion_matrix(true_class, pred_class)


def accuracy(true_class, pred_class):
    return accuracy_score(true_class, pred_class)


def precision(true_class, pred_class):
    return precision_score(true_class, pred_class)


def recall(true_class, pred_class):
    return recall_score(true_class, pred_class)


def f_score(true_class, pred_class):
    return f1_score(true_class, pred_class)
