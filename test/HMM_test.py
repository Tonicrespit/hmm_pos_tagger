from markovify.HMM import HMM
from markovify.ViterbiDecoder import ViterbiDecoder


def HMM_test():
    model = HMM()
    model.train('.', 'test.txt')
    print(model.a)
    print(model.b)

    decoder = ViterbiDecoder(model)
    result = decoder.viterbi(["Janet", "will", "back", "the", "bill"])
    print(result)


HMM_test()
