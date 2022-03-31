from gradientDescent.gradientResult import GradientDescentResult
from gradientDescent.stepCalculators import ConstantStepCalculator, ExponentialStepCalculator, DichotomyStepCalculator, \
    WolfeConditionsStepCalculator
from gradientDescent.stopCriteria import ArgumentStopCriteria, IterationsStopCriteria
from utils.gradientUtils import GradientHelper
from utils.mathUtils import sub, mul


class GradientDescent:
    def __init__(self, stopCriteria, stepCalculator):
        self.stopCriteria = stopCriteria
        self.stepCalculator = stepCalculator

    def __str__(self):
        stop = ''.join(map(str, self.stopCriteria))
        return f'Stop on [{stop}]\nStep is {self.stepCalculator}'

    def execute(self, function, start) -> GradientDescentResult:
        result = GradientDescentResult()
        gradient = GradientHelper.gradient(function)

        current = start
        result.points.append(current)
        while True:
            gradientValues = GradientHelper.gradientValues(gradient, current)
            result.gradientCalls += 1

            result.iterations += 1
            step = self.stepCalculator.calculate(result, function, gradientValues)

            next = sub(current, mul(gradientValues, step))
            result.points.append(next)

            stop = False
            for stopCriteria in self.stopCriteria:
                if stopCriteria.stop(result):
                    stop = True
                    break
            if stop:
                break

            current = next

        return result


def createStopCriteria(eps, maxIterations):
    stopCriteria = [ArgumentStopCriteria(eps)]
    if maxIterations is not None:
        stopCriteria.append(IterationsStopCriteria(maxIterations))
    return stopCriteria


class SimpleGradientDescent(GradientDescent):
    def __init__(self, eps, constantStep, maxIterations=None):
        super().__init__(
            createStopCriteria(eps, maxIterations),
            ConstantStepCalculator(constantStep)
        )


class ExponentialGradientDescent(GradientDescent):
    def __init__(self, eps, initialStep, exp, maxIterations=None):
        super().__init__(
            createStopCriteria(eps, maxIterations),
            ExponentialStepCalculator(initialStep, exp)
        )


class DichotomyGradientDescent(GradientDescent):
    def __init__(self, eps, epsDichotomy, maxStep, maxDichotomyIterations, maxIterations=None):
        super().__init__(
            createStopCriteria(eps, maxIterations),
            DichotomyStepCalculator(maxStep, epsDichotomy, maxDichotomyIterations)
        )


class WolfeGradientDescent(GradientDescent):
    def __init__(self, eps, c1, c2, maxStep, maxWolfeIterations, maxIterations=None):
        super().__init__(
            createStopCriteria(eps, maxIterations),
            WolfeConditionsStepCalculator(c1, c2, maxStep, maxWolfeIterations)
        )
