from markovify.markovify import markovify
from markovify.utils.CrossValidation import cross_val_score

from nltk.corpus import brown


def HMM_test():
    corpus = brown.tagged_sents(categories='news', tagset='universal')
    # mini_corpus = list(corpus[i] for i in range(0, 320))

    m = markovify()
    result = cross_val_score(m, sentences=corpus, verbose=True)
    print(result)


HMM_test()
