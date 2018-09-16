from .Models.HMM import HMM
from .Models.ViterbiDecoder import ViterbiDecoder
from .Utils.CorpusParser import CorpusParser
from .Utils.Utils import get_words

from multiprocessing import Pool


class Markovify:
    def __init__(self, smoothing='max', alpha=None):
        """
        Inits the model.

        :param smoothing: Hyperparamether for the HMM model.
        :param alpha:  Hyperparamether for the HMM model.
        """
        self.hmm = HMM(smoothing=smoothing, alpha=alpha)
        self.decoder = None

    def fit(self, sentences=None, root=None, fileids='.*', encoding='utf8'):
        """
        Fits a HMM to the data. Only a list of sentences or a root directory has to be provided.

        :param sentences: List of sentence. Each sentence is a list of tuples (word, tag)
        :param root: Root folder with the tagged corpora.
        :param fileids: Files to read in the root. By default, '.*' which means 'all'.
        :param encoding: Encoding of the files.
        :return: self (Trained model).
        """
        self.hmm.train(sentences, root, fileids, encoding)
        self.decoder = ViterbiDecoder(self.hmm)
        return self

    def predict(self, words=None, root=None, fileids='.*', encoding='utf8', n_jobs=8):
        """
        Tags a sentence or a set of sentences.

        Only one has to be provided:
            - Sentence.
            - Sentences.
            - Root.

        :param words: List (or list of lists) of tuples (word, tag).
        :param root: Folder with the text files indise.
        :param fileids: Files to read in the root. By default, '.*' which means 'all'.
        :param encoding: Encoding of the files.
        :param n_jobs: Number of CPUs used to compute the predictions.
        :return: List (or list of lists) of tags, deppending on input's structure.
        """
        if self.decoder is None:
            raise NotFittedError("This instance is not fitted yet. Call 'fit' with appropriate arguments before "
                                 "using this method.")
        if words is not None:
            words = get_words(words)
            if type(words[0]) is str:
                tagged = self.decoder.viterbi(words)
            else:
                # tagged = [self.decoder.viterbi(sent) for sent in words]
                with Pool(processes=n_jobs) as pool:
                    tagged = pool.map(self.decoder.viterbi, words)

        elif root is not None:
            reader = CorpusParser(root, fileids, encoding)
            sents = reader.sentences()

            with Pool(processes=n_jobs) as pool:
                tagged = pool.map(self.decoder.viterbi, sents)
            # tagged = [self.decoder.viterbi(sent) for sent in sents]

        return tagged

    def copy(self):
        """
        Make a deep copy of the object.

        :return: Copy.
        """
        hmm = self.hmm.copy()
        decoder = ViterbiDecoder(self.hmm)

        copy = Markovify()
        copy.hmm = hmm
        copy.decoder = decoder
        return copy
