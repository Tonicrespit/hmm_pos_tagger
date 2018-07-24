import pandas as pd
import numpy as np
import json
from . import Tagsets
from ..Utils.CorpusParser import CorpusParser
from ..Utils.Utils import increment


class HMM(object):
    """
    Data structure to represent a Hidden Markov Model
    """

    def __init__(self, q=None, a=None, b=None, smoothing='laplace', alpha=1, tag_count=None):
        """
        Init the data structure.

        :param q: set of states.
        :param a: transition probability matrix. Each a[i, j] represents the probability of moving from state i to j.
        :param b: observation likelihoods. b[tag, word] = likelihood of 'word' being of class 'tag'
        :param smoothing: Smoothing technique. Needed to deal with unknown words.
            - None: No smoothing is used. b[tag, word] = count(word, tag) / count(tag)
            - Laplace: Additive smoothing. b[tag, word] = (count(word, tag) + alpha) / (count(tag) + alpha * size(vocabulary)).
        :param alpha:
        :param tag_count: Tag count. Used for trained Models using LaPlace smoothing.
        """
        if smoothing in ['laplace', 'max', 'none']:
            self._smoothing = smoothing
            self._alpha = alpha
        self.q = q
        self._a = a
        self._b = b
        self.tag_count = tag_count
        self.trained = a is not None and b is not None  # If the user provides a and b, then we already have a model.

    def train(self, sentences=None, root=None, fileids='.*', encoding='utf8'):
        """
        Trains the Hidden Markov Model.

        :param sentences: Tagged sentences. If provided, the others arguments will be ignored.
        :param root: Directory.
        :param fileids: List of files that have to be read. '.*' if all files have to be parsed.
        :param encoding: File encoding. UTF-8 by default.
        """
        if sentences is None and root is None:
            return -1

        bigram_states = {}   # Counts the frequency of two states appearing one after the other.
        tag_word_count = {}  # Counts how many times has a tag.

        if sentences is None:
            reader = CorpusParser(root, fileids, encoding)
            sentences = reader.tagged_sentences()

        for sentence in sentences:
            current = Tagsets.START_TAG

            for word in sentence:
                # Each word is a tuple ("cat", "NN")
                token = word[0]
                tag = word[1]

                last = current  # Last state (t_{i - 1})
                current = tag   # Current state (t_i)

                tag_word_count = increment(tag_word_count, tag, token)
                bigram_states = increment(bigram_states, current, last)

            bigram_states = increment(bigram_states, Tagsets.END_TAG, current)  # Link the last word with the stop tag

        self.q = tuple([Tagsets.START_TAG]) + tuple(bigram_states.keys())
        self._a = self.compute_a(bigram_states)
        self._b = self.compute_b(tag_word_count)
        self.tag_count = self.compute_tag_count(tag_word_count)
        self.trained = True

    def compute_a(self, dictionary):
        """
        Given a dictionary with the bigrams of states, computes the matrix A.

        a[i, j] = p(t_i | t_j) = C(t_j, t_i)/C(t_j)

        Where:
            C(t_j, t_i): How many times t_j is followed by t_i.
            C(t_j): Number of t_j occurrences.

        :param dictionary: Dictionary of bigram states with their count dictionary[s_i][s_j] = count
        :return: transition probability matrix.
        """
        n = len(self.q)
        a = np.zeros((n, n))
        for s_i in self.q:
            for s_j in self.q:
                if s_j in dictionary and s_i in dictionary[s_j]:
                    i = self.q.index(s_i)
                    j = self.q.index(s_j)
                    a[i, j] = dictionary[s_j][s_i] / dictionary[s_j][0]

        return pd.DataFrame(a, columns=self.q, index=self.q)

    def compute_b(self, dictionary):
        """
        Given a dictionary with the count of how many times a word has a tag, computes the matrix B.

        b[w, t] = p(w | t) = C(t, w) / C(t)

        C(t, w): how many times the word w has the tag t.
        C(t): Count of tag t.

        :param dictionary: Dictionary of words and tags counts.
        :return: observation likelihood matrix.
        """
        dict_b = {}  # We temporarily use a dictionary instead of a matrix because we don't have a list of words.
        unique_words = []

        for t in dictionary.keys():
            for w in dictionary[t].keys():
                if w != 0:
                    dict_b[w, t] = dictionary[t][w] / dictionary[t][0]
                    if w not in unique_words:
                        unique_words.append(w)

        rows = len(self.q)
        cols = len(unique_words)
        b = np.zeros((rows, cols))
        for (w, t) in dict_b:
            i = self.q.index(t)
            j = unique_words.index(w)
            if self._smoothing == 'none' or self._smoothing == 'max':
                b[i, j] = dict_b[w, t]
            elif self._smoothing == 'laplace':
                if t in dictionary:
                    count_t = dictionary[t][0]
                else:
                    count_t = 0
                if t in dictionary and w in dictionary[t]:
                    count_t_w = dictionary[t][w]
                else:
                    count_t_w = 0
                b[i, j] = (count_t_w + self._alpha) / (count_t + self._alpha * len(unique_words))

        return pd.DataFrame(b, columns=unique_words, index=self.q)

    def compute_tag_count(self, dictionary):
        """
        Gets the count for each tag.

        :param dictionary: Dictionary of tags.
        :return:
        """
        count = np.zeros(len(self.q))
        for tag in dictionary.keys():
            i = self.q.index(tag)
            count[i] += dictionary[tag][0]
        return count

    def transition_probability(self, s1, s0):
        """
        From matrix a get a get a[s1, s0]

        :param s1: Current state.
        :param s0: Previous state.
        :return: transition probability.
        """
        return self._a.loc[self.q[s1], self.q[s0]]

    def observation_likelihood(self, tag, word):
        """
        From matrix b get b[tag, word]

        :param word: Word.
        :param tag: Tag.
        :return: Observation likelihood.
        """
        if word in self._b.columns:
            return self._b.loc[self.q[tag], word]
        else:
            if self._smoothing == 'laplace':
                return self._alpha / (self.tag_count[tag] + self._alpha * len(self._b.columns))
            elif self._smoothing == 'max':
                most_probable_tag = np.argmax(self.tag_count)
                if tag == most_probable_tag:
                    return 1
                else:
                    return 0
            else:
                return 0

    def to_json(self, file=None):
        """
        Serializes the object to JSON.

        :param file: File where the JSON has to be written. If file == None, the JSON string is returned.
        :return: JSON string.
        """
        data = {
            'a': self._a.to_dict(),
            'b': self._b.to_dict(),
            'smoothing': self._smoothing,
            'alpha': self._alpha,
            'tag_count': self.tag_count.tolist(),
            'q': self.q,
            "trained": self.trained
        }

        if file is not None:
            with open(file, 'w') as outfile:
                json.dump(data, outfile)
        else:
            return json.dumps(response)

    def from_json(self, file):
        """
        Given a file with a JSON dump, retrieves the object.

        :param file: Path to the file.
        """
        with open(file, 'r') as infile:
            dict = json.load(infile)
            self.q = tuple(dict['q'])
            self._smoothing = dict['smoothing']
            self._alpha = dict['alpha']
            self.tag_count = np.array(dict['tag_count'])
            self._a = pd.DataFrame.from_dict(loaded['a'])
            self._b = pd.DataFrame.from_dict(loaded['b'])
            self.trained = dict['trained']

    def copy(self):
        """
        Make a deep copy of the object.

        :return: Copy.
        """
        a, b, q, tag_count = None, None, None, None

        if self._a is not None:
            a = self._a.copy()
        if self._b is not None:
            b = self._b.copy()
        if self.q is not None:
            q = self.q
        if self.tag_count is not None:
            tag_count = np.copy(self.tag_count)
        smoothing = self._smoothing
        alpha = self._alpha

        return HMM(q, a, b, smoothing, alpha, tag_count)



