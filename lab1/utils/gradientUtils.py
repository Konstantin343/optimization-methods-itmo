import sympy as sp


class GradientHelper:
    @staticmethod
    def gradient(func):
        res = {}
        for x in func.free_symbols:
            res[x] = sp.diff(func, x)
        return res

    @staticmethod
    def gradientValues(grad, xs):
        res = {}
        for x in grad.keys():
            res[x] = grad[x].evalf(subs=xs)
        return res
