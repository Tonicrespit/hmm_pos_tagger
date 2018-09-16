import numpy as np
from nltk.corpus import brown

from HMM.Markovify import *
from HMM.Utils.CrossValidation import cross_validation, cross_validation_score
from HMM.Utils.Scoring import *


def hmm_test():
    corpus = brown.tagged_sents(categories='adventure', tagset='universal')
    corpus_news = brown.tagged_sents(categories='news', tagset='universal')
    corpus_religion = brown.tagged_sents(categories='religion', tagset='universal')
    corpus_romance = brown.tagged_sents(categories='romance', tagset='universal')
    # mini_corpus = list(corpus[i] for i in range(0, 200))

    m_max = Markovify(smoothing='max')
    acc, m_max = cross_validation_score(m_max, sentences=corpus, verbose=False, cv=10, return_model=True)
    print(acc[np.argmax([i['accuracy'] for i in acc])])

    predicted_corpus = m_max.predict(corpus_news)
    print({'accuracy': accuracy(corpus_news, predicted_corpus),
           'precision': precision(corpus_news, predicted_corpus),
           'recall': recall(corpus_news, predicted_corpus)})

    predicted_corpus = m_max.predict(corpus_religion)
    print({'accuracy': accuracy(corpus_religion, predicted_corpus),
           'precision': precision(corpus_religion, predicted_corpus),
           'recall': recall(corpus_religion, predicted_corpus)})

    predicted_corpus = m_max.predict(corpus_romance)
    print({'accuracy': accuracy(corpus_romance, predicted_corpus),
           'precision': precision(corpus_romance, predicted_corpus),
           'recall': recall(corpus_romance, predicted_corpus)})

    # m_laplace = Markovify(smoothing='laplace', alpha=1)
    # scores_laplace = cross_validation_score(m_laplace, sentences=corpus, verbose=False, cv=10)
    # print(scores_laplace)

    # amax_max = np.argmax([i['accuracy'] for i in scores_max])
    # alaplace_max = np.argmax([i['accuracy'] for i in scores_laplace])

    # print(scores_max[amax_max])
    # print(scores_laplace[alaplace_max])

    # m = m.fit(mini_corpus)
    # pred = m.predict(mini_corpus)
    # print(confusion_matrix(mini_corpus, pred))
    # print(m.predict(mini_corpus))
    # print(m.predict([['The', 'black'], ['man', 'is']]))

    # corpus = brown.tagged_sents(categories='news', tagset='universal')

    # model_laplace = Markovify(smoothing='laplace', alpha=1)
    # start = time.time()
    # brown_news_accuracy_laplace = cross_validation_score(model_laplace, sentences=corpus, verbose=True)
    # end = time.time()
    # print('Time elpased: {}s.'.format(end - start))
    # print('Best accuracy for Laplace smoothing: {}.'.format(np.max(brown_news_accuracy_laplace)))
    # tagged = model.predict(['The', 'black', 'man'])
    # print(result)
    # print(tagged)
    # print(confusion_matrix(corpus, corpus))
    # print(accuracy(corpus, corpus))
    # print(recall(corpus, corpus))
    # print(precision(corpus, corpus))
    # print(f_score(corpus, corpus))


if __name__ == '__main__':
    hmm_test()
