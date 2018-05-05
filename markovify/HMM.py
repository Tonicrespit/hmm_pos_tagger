import pandas as pd
import numpy as np
import json
import markovify.Tagsets as Tagsets
from markovify.CorpusParser import CorpusParser
from markovify.Utils import increment, dict_to_matrix


class HMM(object):
    """
    Data structure to represent a Hidden Markov Model
    """

    def __init__(self, q=None, a=None, b=None):
        """
        Init the data structure.

        Keyword arguments:
            q: set of states.
            a: transition probability matrix. Each a[i, j] represents the probability of moving from state i to state j.
            b: observation likelihoods. b[tag, word] = likelihood of 'word' being of class 'tag'
        """
        self.q = q
        self.a = a
        self.b = b
        self.trained = a is not None and b is not None  # If the user provides a and b, then we already have a model.

    def train(self, root, fileids='.*', encoding='utf8'):
        """
        Trains the Hidden Markov Model.

        :param root: Directory.
        :param fileids: List of files that have to be read. '.*' if all files have to be parsed.
        :param encoding: File encoding. UTF-8 by default.
        """
        bigram_states = {}   # Counts the frequency of two states appearing one after the other.
        tag_word_count = {}  # Counts how many times has a tag.

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

        self.q = tuple([Tagsets.START_TAG]) + tuple(bigram_states.keys())
        self.a = self.compute_a(bigram_states)
        self.b = self.compute_b(tag_word_count)
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
                dict_b[w, t] = dictionary[t][w] / dictionary[t][0]
                if w not in unique_words:
                    unique_words.append(w)

        rows = len(self.q)
        cols = len(unique_words)
        b = np.zeros((rows, cols))
        for (w, t) in dict_b:
            i = self.q.index(t)
            j = unique_words.index(w)
            b[i, j] = dict_b[w, t]

        return pd.DataFrame(b, columns=unique_words, index=self.q)

    def to_json(self, file=None):
        """
        Serializes the object to JSON.

        :param file: File where the JSON has to be written. If file == None, the JSON string is returned.
        :return: JSON string.
        """
        response = {
            'a': self.a.to_dict(),
            'b': self.b.to_dict(),
            'q': self.q, "trained": self.trained
        }

        if file is not None:
            with open(file, 'w') as outfile:
                json.dump(response, outfile)
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
            self.a = pd.DataFrame.from_dict(loaded['a'])
            self.b = pd.DataFrame.from_dict(loaded['b'])
            self.trained = dict['trained']

