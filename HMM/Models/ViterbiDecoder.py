import numpy as np
from . import Tagsets


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
        path_probabilities = np.zeros((len(self.hmm.q), len(sentence) + 1))
        backpointers = np.zeros((len(self.hmm.q), len(sentence) + 1))
        for s in range(0, len(self.hmm.q)):
            s0 = self.hmm.q.index(Tagsets.START_TAG)
            w = sentence[0]
            path_probabilities[s, 0] = self.hmm.transition_probability(s0, s) * self.hmm.observation_likelihood(s, w)
            backpointers[s, 0] = 0

        if len(sentence) > 1:
            for t in range(1, len(sentence)):
                for s in range(0, len(self.hmm.q)):
                    path_probabilities[s, t] = self._best_previous_path(path_probabilities[:, t - 1], s, sentence[t])
                    backpointers[s, t] = self._get_backpointer(path_probabilities[:, t - 1], s)

        t = len(sentence)
        end_tag_index = self.hmm.q.index(Tagsets.END_TAG)
        path_probabilities[end_tag_index, t] = self._best_previous_path(path_probabilities[:, t - 1],
                                                                        end_tag_index,
                                                                        None)
        backpointers[end_tag_index, t] = self._get_backpointer(path_probabilities[:, t - 1],
                                                               end_tag_index)
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
        values = np.zeros(len(path_probabilities))
        for s2 in range(0, len(self.hmm.q)):
            if o is not None:
                values[s2] = path_probabilities[s2] * self.hmm.transition_probability(s2, s) * self.hmm.observation_likelihood(s, o)
            else:
                values[s2] = path_probabilities[s2] * self.hmm.transition_probability(s2, s)

        return np.max(values)

    def _get_backpointer(self, path_probabilities, s):
        """
        Gets the best next tag to add to the path of tags:
            state s' that maximizes path_probabilities[s'] * a[s', s]

        :param path_probabilities: Vector of length len(q) with the path probabilities.
        :return: Tag that maximizes the path probability
        """
        values = np.zeros(len(path_probabilities))
        for s2 in range(0, len(self.hmm.q)):
            values[s2] = path_probabilities[s2] * self.hmm.transition_probability(s2, s)

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
            tags.append(self.hmm.q[pointer])
            pointer = col[pointer].astype(int)

            ncol -= 1

        if Tagsets.END_TAG in tags:
            tags.remove(Tagsets.END_TAG)
        if Tagsets.START_TAG in tags:
            tags.remove(Tagsets.START_TAG)
        tags.reverse()
        return tags
