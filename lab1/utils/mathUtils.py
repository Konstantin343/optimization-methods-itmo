import math


def div(xs1, n):
    xs = {}
    for x in xs1:
        xs[x] = xs1[x] / n
    return xs


def mul(xs1, n):
    xs = {}
    for x in xs1:
        xs[x] = xs1[x] * n
    return xs


def add(xs1, xs2):
    xs = {}
    for x in xs1:
        if x in xs2:
            xs[x] = xs1[x] + xs2[x]
        else:
            xs[x] = xs1[x]
    return xs


def sub(xs1, xs2):
    xs = {}
    for x in xs1:
        if x in xs2:
            xs[x] = xs1[x] - xs2[x]
        else:
            xs[x] = xs1[x]
    return xs


def dot(xs1, xs2):
    res = 0.0
    for x in xs1:
        if x in xs2:
            res += xs1[x] * xs2[x]
    return res


def norm(xs1, xs2):
    res = 0.0
    for x in xs1:
        if x in xs2:
            res += (xs1[x] - xs2[x]) ** 2
        else:
            res += xs1[x] ** 2
    return math.sqrt(res)
