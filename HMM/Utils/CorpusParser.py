from nltk.corpus.reader import TaggedCorpusReader


class CorpusParser:
    def __init__(self, root, fileids='.*', encoding='utf8'):
        """
        Reads all the files in root.

        :param root: Directory.
        :param fileids: List of files that have to be read. '.*' if all files have to be parsed.
        :param encoding: File enconding
        """
        self._reader = TaggedCorpusReader(root, fileids, encoding=encoding)

    def words(self):
        """
        Returns all the words in the corpora.

        :return: List of words.
        """
        return self._reader.words()

    def tagged_words(self):
        """
        Returns all words of the corpora with their corresponding tag.

        :return: List of tuples (word, tag)
        """
        return self._reader.tagged_words()

    def sentences(self):
        """
        Returns a list of all sentences.

        :return: List of lists of words. Each list represents a sentence, with a list of its words in it.
        """
        return self._reader.sents()

    def tagged_sentences(self):
        """
        Returns a list of all sentences with the tag of each word.

        :return: List of lists of tuples. Each sentence is a list with all its members being tuples (word, tag).
        """
        return self._reader.tagged_sents()
