import numpy as np
import sympy as sp
import pandas as pd
from matplotlib import pyplot as plt

from gradientDescent.gradientDescents import *
from utils.generatorsUtils import generateFunction


# region Plots

def f1(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


def f2(x1, x2):
    return x1 ** 2 - 2 * x1 * x2 + 2 * x2 ** 2 - x1 + x2 - 2


def f3(x1, x2):
    return (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2


plotFunctions = [
    (f1, [2, 3], np.mgrid[-5:6, -5:6]),
    (f2, [4, -3], np.mgrid[-5:6, -5:6]),
    (f3, [-4, 4], np.mgrid[-5:6, -5:6])
]
plotAlgorithms = [
    SimpleGradientDescent(1e-5, 1e-2),
    ExponentialGradientDescent(1e-5, 1e-2, 1.01),
    DichotomyGradientDescent(1e-5, 1e-3, 1, 1000),
    WolfeGradientDescent(1e-5, 0.01, 1, 1e-1, 100)
]


def drawPlot(f, start, bounds, algorithm):
    x1, x2 = sp.symbols('x1 x2')
    func = f(x1, x2)

    x, y = bounds
    z = f(x, y)
    fig, ax = plt.subplots()
    c = ax.contour(x, y, z)
    ax.clabel(c)

    result = algorithm.execute(func, {x1: start[0], x2: start[1]})

    t1, t2 = [], []
    for p in result.points:
        t1.append(p[x1])
        t2.append(p[x2])
    ax.plot(t1, t2, color='r')
    ax.scatter(start[0], start[1], color='r')
    ax.scatter(result.points[-1][x1], result.points[-1][x2], color='g')

    ax.grid()
    plt.title(f'{algorithm}\n{f.__name__}{x1, x2} = {func}', fontsize=10)
    plt.show()

    print('argMin =', result.points[-1])
    print('min = ', func.evalf(subs=result.points[-1]))
    print('iterations = ', result.iterations)
    print('functionCalls = ', result.functionCalls)
    print('gradientCalls = ', result.gradientCalls)
    print('=' * 50)


def drawPlots():
    for (f, start, bounds) in plotFunctions:
        for algorithm in plotAlgorithms:
            drawPlot(f, start, bounds, algorithm)


# endregion


def buildTable(ns, ks, algorithms):
    data = dict(map(lambda a: (a, np.ndarray((len(ns), len(ks)))), algorithms))
    for (i, n) in enumerate(ns):
        for (j, k) in enumerate(ks):
            f = generateFunction(n, k)
            start = {}
            for x in f.free_symbols:
                start[x] = 5
            for algorithm in algorithms:
                result = algorithm.execute(f, start)
                data[algorithm][i][j] = int(result.iterations)
                print(n, k, result.iterations)
    for algorithm in algorithms:
        dataFrame = pd.DataFrame(data=data[algorithm], columns=ks, index=ns)
        print(dataFrame)
        plt.title(str(algorithm))
        for i in range(len(data[algorithm])):
            plt.plot(ks, data[algorithm][i], label=f'n = {ns[i]}', marker='o')
        plt.grid()
        plt.legend()
        plt.show()


def buildTables():
    buildTable(
        [2, 3, 4, 5],
        [1, 200, 400, 600, 800, 1000, 1200],
        [WolfeGradientDescent(1e-5, 0.01, 1, 1e-1, 100), DichotomyGradientDescent(1e-6, 1e-3, 1, 1000)]
    )


def main():
    drawPlots()
    buildTables()


if __name__ == '__main__':
    main()
