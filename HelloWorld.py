from HMM.Markovify import Markovify
from nltk.corpus import brown

def hmm_test():
    training_corpus = brown.tagged_sents(categories='adventure', tagset='universal')

    model = Markovify(smoothing='max')
    model = model.fit(training_corpus)
    tagged_test = model.predict(['My', 'first', 'sentence'])
    print(tagged_test)


if __name__ == '__main__':
    hmm_test()
