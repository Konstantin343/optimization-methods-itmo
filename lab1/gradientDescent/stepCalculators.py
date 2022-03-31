from utils.gradientUtils import GradientHelper
from utils.mathUtils import sub, mul, dot


class StepCalculator:
    def calculate(self, currentResult, function, gradientValues):
        pass


class ConstantStepCalculator(StepCalculator):
    def __init__(self, constantStep):
        self.constantStep = constantStep

    def __str__(self):
        return f'const={self.constantStep}'

    def calculate(self, currentResult, function, gradientValues):
        return self.constantStep


class ExponentialStepCalculator(StepCalculator):
    def __init__(self, constantStep, exp):
        self.initial = constantStep
        self.step = constantStep
        self.exp = exp

    def __str__(self):
        return f'exp({self.exp},iters)*{self.initial}'

    def calculate(self, currentResult, function, gradientValues):
        self.step *= self.exp
        return self.step


class DichotomyStepCalculator(StepCalculator):

    def __init__(self, maxStep, eps, maxIterations):
        self.__K = 2.5
        self.maxStep = maxStep
        self.eps = eps
        self.maxIterations = maxIterations
        self.delta = eps / self.__K

    def __str__(self):
        return f'dichotomy(0<=a<={self.maxStep})'

    def calculate(self, currentResult, function, gradientValues):
        def func(x):
            currentResult.functionCalls += 1
            return function.evalf(subs=sub(currentResult.points[-1], mul(gradientValues, x)))

        iterations = 0
        a, b = 0.0, self.maxStep
        while abs(b - a) > self.eps and iterations < self.maxIterations:
            iterations += 1

            x1 = (a + b) / 2 - self.delta
            x2 = (a + b) / 2 + self.delta

            if func(x1) >= func(x2):
                a = x1
            else:
                b = x2

        return (a + b) / 2


class WolfeConditionsStepCalculator(StepCalculator):
    def __str__(self):
        return f'wolfe(c1={self.c1},c2={self.c2})'

    def __init__(self, c1, c2, maxStep, maxIterations):
        self.c1 = c1
        self.c2 = c2
        self.maxStep = maxStep
        self.maxIterations = maxIterations

    def calculate(self, currentResult, function, gradientValues):
        def func(x):
            currentResult.functionCalls += 1
            return function.evalf(subs=sub(currentResult.points[-1], mul(gradientValues, x)))

        iterations = 0
        a, b = 0.0, self.maxStep
        fxOld = func(0)
        while iterations < self.maxIterations:
            iterations += 1

            xNew = (a + b) / 2
            fx = func(xNew)

            newGradient = GradientHelper.gradientValues(
                GradientHelper.gradient(function),
                sub(currentResult.points[-1], mul(gradientValues, xNew))
            )
            currentResult.gradientCalls += 1

            wolfeCond1 = (fx <= fxOld + self.c1 * xNew * dot(gradientValues, gradientValues))
            wolfeCond2 = (abs(dot(newGradient, gradientValues)) <= abs(self.c2 * dot(gradientValues, gradientValues)))

            if wolfeCond1 and wolfeCond2:
                break
            elif wolfeCond1:
                a = xNew
            else:
                b = xNew

        return (a + b) / 2
