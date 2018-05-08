from markovify.HMM import HMM


def HMM_test():
    model = HMM()
    model.train('./test/', 'test.txt')
    print(model.a)
    print(model.b)


HMM_test()
