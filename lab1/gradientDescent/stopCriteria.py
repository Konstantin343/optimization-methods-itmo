import numpy as np
from utils.mathUtils import sub


class StopCriteria:
    def stop(self, currentResult):
        pass


class IterationsStopCriteria(StopCriteria):
    def __init__(self, iterations):
        self.iterations = iterations

    def __str__(self):
        return f'iters>={self.iterations}'

    def stop(self, currentResult):
        return currentResult.iterations >= self.iterations


class ArgumentStopCriteria(StopCriteria):
    def __init__(self, eps):
        self.eps = eps

    def __str__(self):
        return f'||Xk - Xk-1||<={self.eps}'

    def stop(self, currentResult):
        values = list(map(
            lambda x: float(x),
            sub(currentResult.points[-2], currentResult.points[-1]).values()
        ))
        return np.linalg.norm(values) < self.eps
