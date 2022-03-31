import random
import numpy as np
import sympy as sp


def generateFunction(n: int, k: float):
    values = [random.uniform(1.0, k) for _ in range(n - 2)]
    values.extend([1.0, k])
    values.sort()
    d = np.diag(values)
    m = np.random.rand(n, n)
    q, _ = np.linalg.qr(m)
    matrix = np.matmul(np.matmul(q, d), np.transpose(q))
    xs = sp.symbols(' '.join(map(lambda x: 'x' + str(x), range(1, n + 1))))
    return sum(matrix[i][j] * xs[i] * xs[j] for i in range(n) for j in range(n))


if __name__ == '__main__':
    print(generateFunction(3, 3))
