import numpy as np
import os

SYNTHETIC_STATES = ("__BEGIN__", "__END__")
PEN_TREEBANK_TAGSET = ('CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
                       'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
                       'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '$', '#', '"', '(', ')', "'", '.', ':')


class HMM(object):
    """
    Data structure to represent a Hidden Markov Model
    """

    def __init__(self, q=PEN_TREEBANK_TAGSET, a=None):
        """
        Init the data structure.

        Keyword arguments:
            q: set of states.
            a: transition probability matrix. Each a[i, j] represents the probability of moving from state i to state j.
        """
        self.q = SYNTHETIC_STATES + tuple(q)
        self.n = len(self.q)
        self.trained = a is not None  # If there is a parameter a, then we already have a trained model

        if a is None:
            self.a = np.zeros((self.n, self.n))
        else:
            self.a = a

    def train(self, corpus):
        """
        Trains the Hidden Markov Model.

        :param corpus: Path to a folder or to a file. If a folder is specified, the model will be trained using every
                       *.txt file in it.
        """
        if os.path.isdir(corpus):
            # Train with all files in dir

        else:
            # Train with a single file


