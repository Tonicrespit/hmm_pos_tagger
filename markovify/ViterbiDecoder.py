import numpy as np
import markovify.Tagsets as Tagsets


class ViterbiDecoder:
    def __init__(self, hmm):
        """

        :param hmm: Trained Hidden Markov Model
        """
        self.hmm = hmm

    def viterbi(self, sentence):
        """
        Using the traditional algorithm of Viterbi to get the most probable tag sequence for a sentence.

        :param sentence: List of words.
        :return: List of tags.
        """
        a = self.hmm.a
        b = self.hmm.b
        q = self.hmm.q

        path_probabilities = np.zeros((len(q), len(sentence) + 1))
        backpointers = np.zeros((len(q), len(sentence) + 1))
        for s in range(0, len(q)):
            path_probabilities[s, 0] = a.loc['<s>', q[s]] * b.loc[q[s], sentence[0]]
            backpointers[s, 0] = 0

        if len(sentence) > 1:
            for t in range(1, len(sentence)):
                for s in range(0, len(q)):
                    path_probabilities[s, t] = self._best_previous_path(path_probabilities[:, t - 1], s, sentence[t])
                    backpointers[s, t] = self._get_backpointer(path_probabilities[:, t - 1], s)

        t = len(sentence)
        path_probabilities[q.index(Tagsets.END_TAG), t] = self._best_previous_path(path_probabilities[:, t - 1],
                                                                                   q.index(Tagsets.END_TAG),
                                                                                   None)
        backpointers[q.index(Tagsets.END_TAG), t] = self._get_backpointer(path_probabilities[:, t - 1],
                                                                          q.index(Tagsets.END_TAG))
        backtrace = self._get_best_path(backpointers)
        return backtrace

    def _best_previous_path(self, path_probabilities, s, o):
        """
        Gets the probability of the most probable path that has gotten us to state s:
            probability of a given state s' that maximizes path_probabilities[s'] * a[s', s] * b[s, o]

        :param path_probabilities: Vector of length len(q) with the path probabilities.
        :param s: Current state.
        :param o: Current word.
        :return: Maximum path probability when adding s to the tags
        """
        a = self.hmm.a
        b = self.hmm.b
        q = self.hmm.q

        values = np.zeros(len(path_probabilities))
        for s2 in range(0, len(q)):
            if o is not None:
                values[s2] = path_probabilities[s2] * a.loc[q[s2], q[s]] * b.loc[q[s], o]
            else:
                values[s2] = path_probabilities[s2] * a.loc[q[s2], q[s]]

        return np.max(values)

    def _get_backpointer(self, path_probabilities, s):
        """
        Gets the best next tag to add to the path of tags:
            state s' that maximizes path_probabilities[s'] * a[s', s]

        :param path_probabilities: Vector of length len(q) with the path probabilities.
        :return: Tag that maximizes the path probability
        """
        a = self.hmm.a
        q = self.hmm.q

        values = np.zeros(len(path_probabilities))
        for s2 in range(0, len(q)):
            values[s2] = path_probabilities[s2] * a.loc[q[s2], q[s]]

        return np.argmax(values)

    def _get_best_path(self, backpointers):
        """
        Given a matrix of backpointers, gets the path of tags with maximum probability.

        :param backpointers: Matrix computed by the Viterbi algorithm.
        :return: List of tags.
        """
        tags = []

        ncol = len(backpointers[0]) - 1
        col = backpointers[:, ncol]
        pointer = np.argmax(col).astype(int)
        while ncol >= 0:
            col = backpointers[:, ncol]
            if self.hmm.q[pointer] != Tagsets.END_TAG or self.hmm.q[pointer] != Tagsets.START_TAG:
                tags.append(self.hmm.q[pointer])
            pointer = col[pointer].astype(int)

            ncol -= 1

        tags.reverse()
        return tags
