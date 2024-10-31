from functools import reduce


class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        print('training start...')
        for i in range(iteration):
            for (input_vec, label) in zip(input_vecs, labels):
                output = self.predict(input_vec)
                print("label=%d, output=%d" % (label, output))
                self._update_weights(input_vec, output, label, rate)
        print('training end.\n')

    def predict(self, input_vec):
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda x: x[0] * x[1], zip(input_vec, self.weights)),
                   0.0)
            + self.bias)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = list(map(
            lambda x: x[1] + rate * delta * x[0],
            zip(input_vec, self.weights)))
        self.bias = self.bias + rate * delta


def sgn(x):
    return 1 if x > 0 else 0


def linear(x):
    return x


def test_perceptron():
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    labels = [1, 0, 0, 0]

    print('Test Perceptron')
    p = Perceptron(2, sgn)
    p.train(input_vecs, labels, 10, 0.1)

    # print weights anf bias
    print(p)
    # test and print result
    print("1 and 1 = %d" % p.predict([1, 1]))
    print("0 and 0 = %d" % p.predict([0, 0]))
    print("1 and 0 = %d" % p.predict([1, 0]))
    print("0 and 1 = %d" % p.predict([0, 1]))


def test_linear_unit():
    # ads fee: 4 8  9  8  7  12 6  10 6  9
    # sales:   9 20 22 15 17 23 18 25 10 20
    print('Test Linear Unit')
    input_vecs = [[4], [8], [9], [8], [7], [12], [6], [10], [6], [9]]
    labels = [9, 20, 22, 15, 17, 23, 18, 25, 10, 20]

    p = Perceptron(1, linear)
    p.train(input_vecs, labels, 10, 0.01)

    # print weights anf bias
    print(p)
    # test and print result
    print("ads fee 5, sales = %d" % p.predict([5]))
    print("ads fee 3, sales = %d" % p.predict([3]))
    print("ads fee 8, sales = %d" % p.predict([8]))
    print("ads fee 15, sales = %d" % p.predict([15]))

if __name__ == '__main__':
    test_perceptron()
    test_linear_unit()