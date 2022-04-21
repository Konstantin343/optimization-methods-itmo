import math
import random
import random as rnd
from datetime import datetime

import memory_profiler
import numpy as np
from matplotlib import pyplot as plt


# region Datasets

def read_dataset(filename):
    file = open(filename, 'r')
    try:
        lines = file.readlines()
        size = int(lines[0])
        train = []
        for i in range(size):
            train.append(list(map(float, lines[i + 1].split())))
        return train
    finally:
        file.close()


def minimax(dataset):
    min_max = [(v, v) for v in dataset[0]]
    for obj in dataset:
        for i in range(len(obj)):
            min_max[i] = (min(min_max[i][0], obj[i]), max(min_max[i][1], obj[i]))
    return min_max


def normalize_dataset(dataset, min_max):
    for obj in dataset:
        for i in range(len(obj)):
            if min_max[i][1] == min_max[i][0]:
                obj[i] = 1.0
            else:
                obj[i] = (obj[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])


def denormalize_coefficients(coefficients, min_max):
    coefficients[0] -= sum(
        [(coefficients[i] * min_max[i - 1][0]) / (min_max[i - 1][1] - min_max[i - 1][0])
         if min_max[i - 1][1] != min_max[i - 1][0] else coefficients[i] for i in range(1, len(coefficients))])
    for i in range(1, len(coefficients)):
        if min_max[i - 1][1] != min_max[i - 1][0]:
            coefficients[i] /= (min_max[i - 1][1] - min_max[i - 1][0])

    for i in range(len(coefficients)):
        coefficients[i] *= (min_max[-1][1] - min_max[-1][0])

    coefficients[0] += min_max[-1][0]


# endregion

# region Common

def predict(obj, coefficients):
    if type(coefficients[0]) is float:
        y = coefficients[0]
    else:
        y = coefficients[0].copy()
    for i in range(len(obj) - 1):
        y += coefficients[i + 1] * obj[i]
    return y


def mse(coefficients, objects):
    result = 0.0
    for obj in objects:
        predicted = predict(obj, coefficients)
        result += (predicted - obj[-1]) ** 2
    return result / len(objects)


def gradient_mse(coefficients, objects):
    gradient = [0 for _ in range(-1, len(objects[0]) - 1)]
    for obj in objects:
        predicted = predict(obj, coefficients)
        for i in range(-1, len(obj) - 1):
            gradient[i + 1] += 2 * (predicted - obj[-1]) * (obj[i] if i >= 0 else 1)
    for i in range(-1, len(objects[0]) - 1):
        gradient[i + 1] /= len(objects)
    return gradient


def exponential_moving_average(q_last, ema_coefficient, q_one):
    return q_one * ema_coefficient + (1 - ema_coefficient) * q_last


def generate_coefficients(features):
    return [rnd.uniform(-1 / (2 * features), 1 / (2 * features)) for _ in range(features)]


# endregion

# region GradientDescent

class StochasticGradientDescent:
    def __init__(self, limit, eps, ema_coef, batch_size):
        self.limit = limit
        self.eps = eps
        self.ema_coef = ema_coef
        self.batch_size = batch_size

    def make_step(self, coefficients, step, gradients, current_objects):
        gradient = gradient_mse(coefficients, current_objects)
        for j in range(len(coefficients)):
            coefficients[j] = coefficients[j] - step[j] * gradient[j]
        return gradient

    def create_step(self, i, coefficients, gradients):
        return [1e-2 for _ in range(len(coefficients))]

    def execute(self, objects, start):
        coefficients = start.copy()

        steps = [coefficients.copy()]
        gradients = [[0.0 for _ in range(len(start))]]
        q = 0.0

        for i in range(self.limit):
            current_objects = []
            sample = range(0, len(objects))
            if self.batch_size is not None:
                sample = rnd.sample(sample, self.batch_size)
            for j in sample:
                current_objects.append(objects[j])

            step = self.create_step(i, coefficients, gradients)
            gradient = self.make_step(coefficients, step, gradients, current_objects)

            steps.append(coefficients.copy())
            gradients.append(gradient.copy())

            # q = exponential_moving_average(q, self.ema_coef, mse(coefficients, current_objects))
            q = np.linalg.norm(np.array(steps[-1]) - np.array(steps[-2]))
            if q < self.eps:
                break

        return steps, coefficients


class MomentumGradientDescent(StochasticGradientDescent):
    def __init__(self, limit, eps, ema_coef, batch_size, gamma):
        super().__init__(limit, eps, ema_coef, batch_size)
        self.gamma = gamma

    def make_step(self, coefficients, step, gradients, current_objects):
        gradient = gradient_mse(coefficients, current_objects)
        for j in range(len(coefficients)):
            previous = gradients[-1][j] * self.gamma
            coefficients[j] = coefficients[j] - (previous + step[j] * gradient[j])
        return gradient


class NesterovGradientDescent(StochasticGradientDescent):
    def __init__(self, limit, eps, ema_coef, batch_size, gamma):
        super().__init__(limit, eps, ema_coef, batch_size)
        self.gamma = gamma

    def make_step(self, coefficients, step, gradients, current_objects):
        new_coefficients = coefficients.copy()
        for j in range(len(coefficients)):
            new_coefficients[j] -= gradients[-1][j] * self.gamma
        new_gradient = gradient_mse(new_coefficients, current_objects)
        for j in range(len(coefficients)):
            previous = gradients[-1][j] * self.gamma
            coefficients[j] = coefficients[j] - (previous + step[j] * new_gradient[j])
        return new_gradient


class AdagradGradientDescent(StochasticGradientDescent):
    def __init__(self, limit, eps, ema_coef, batch_size):
        super().__init__(limit, eps, ema_coef, batch_size)
        self.G = None

    def create_step(self, i, coefficients, gradients):
        if len(gradients) == 1:
            return super().create_step(i, coefficients, gradients)
        if self.G is None:
            self.G = [0.0 for _ in range(len(gradients[0]))]
        for k in range(len(gradients[-1])):
            self.G[k] += gradients[-1][k] ** 2
        result = []
        for g in self.G:
            result.append(1e-1 / math.sqrt(g + 1e-8))
        return result


class RMSPropGradientDescent(StochasticGradientDescent):
    def __init__(self, limit, eps, ema_coef, batch_size, beta):
        super().__init__(limit, eps, ema_coef, batch_size)
        self.beta = beta
        self.ema_grad = None

    def make_step(self, coefficients, step, gradients, current_objects):
        gradient = gradient_mse(coefficients, current_objects)

        if self.ema_grad is None:
            self.ema_grad = [0.0 for _ in range(len(coefficients))]

        for j in range(len(coefficients)):
            self.ema_grad[j] = exponential_moving_average(self.ema_grad[j], self.beta, gradient[j] ** 2)

        for j in range(len(coefficients)):
            coefficients[j] = coefficients[j] - (step[j] * gradient[j] / (math.sqrt(self.ema_grad[j]) + 1e-8))
        return gradient


class AdamGradientDescent(StochasticGradientDescent):
    def __init__(self, limit, eps, ema_coef, batch_size, beta1, beta2):
        super().__init__(limit, eps, ema_coef, batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.ema_grad = None
        self.ema_grad_sqr = None

    def make_step(self, coefficients, step, gradients, current_objects):
        gradient = gradient_mse(coefficients, current_objects)

        if self.ema_grad is None:
            self.ema_grad = [0.0 for _ in range(len(coefficients))]
        if self.ema_grad_sqr is None:
            self.ema_grad_sqr = [0.0 for _ in range(len(coefficients))]

        p = len(gradients)
        for j in range(len(coefficients)):
            self.ema_grad[j] = exponential_moving_average(self.ema_grad[j], self.beta1, gradient[j])
            self.ema_grad_sqr[j] = exponential_moving_average(self.ema_grad_sqr[j], self.beta2, gradient[j] ** 2)

        for j in range(len(coefficients)):
            m = self.ema_grad[j] / (1 - math.pow(self.beta1, p))
            v = self.ema_grad_sqr[j] / (1 - math.pow(self.beta2, p))
            coefficients[j] = coefficients[j] - (step[j] * m / (math.sqrt(v) + 1e-8))
        return gradient


# endregion

# region Plots

def get_dataset_file(number):
    return f'datasets/{number}.txt'


def draw_contour(title, steps, min_max, old_dataset):
    xs, ys = [], []
    for point in steps:
        denormalize_coefficients(point, min_max)
        xs.append(point[0])
        ys.append(point[1])

    def update_bound(bound, arg):
        return (bound * 2) if (bound * arg < 0) else (bound / 2)

    lx = update_bound(min(xs), 1)
    rx = update_bound(max(xs), -1)
    ly = update_bound(min(ys), 1)
    ry = update_bound(max(ys), -1)
    x, y = np.mgrid[lx:rx, ly:ry]
    f = mse([x, y], old_dataset)
    plt.title(title)
    plt.xlabel('c0')
    plt.ylabel('c1')
    plt.contour(x, y, f)
    plt.plot(xs, ys)
    plt.show()


def draw_points(title, dataset, coefficients):
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(
        list(map(lambda o: o[0], dataset)),
        list(map(lambda o: o[1], dataset))
    )

    xmax = max(map(lambda o: o[0], dataset))
    plt.plot([0, xmax], [coefficients[0], coefficients[0] + xmax * coefficients[1]], color='orange')
    plt.show()


def create_dataset():
    dataset = []
    file = open('datasets/tmp.txt', 'w')
    file.write('500\n')
    for i in range(500):
        x = i / 3
        y = float(random.randint(0 if i % 5 == 0 else (i // 2), i))
        dataset.append([x, y])
        file.write(f'{x} {y}\n')

    return dataset


# endregion

# region Tasks

LIMIT = 30000
EMA_COEFFICIENT = 0.1
EPS = 1e-4


def task_1():
    algorithms = [
        ('SGD', StochasticGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, 1)),
        ('Minibatch GD 32', StochasticGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, 32)),
        ('GD', StochasticGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, None)),
    ]

    dataset = create_dataset()
    min_max = minimax(dataset)
    old_dataset = []
    for obj in dataset:
        old_dataset.append(obj.copy())
    normalize_dataset(dataset, min_max)

    start = generate_coefficients(len(dataset[0]))
    for (name, gradientDescent) in algorithms:
        for normalize in [True, False]:
            if normalize:
                steps, coefficients = gradientDescent.execute(dataset, start)
                denormalize_coefficients(coefficients, min_max)
                title = ' (with normalization)'
            else:
                steps, coefficients = gradientDescent.execute(old_dataset, start)
                title = ''
            draw_points(f'{name}{title}\n{len(steps)} iterations', old_dataset, coefficients)
            draw_contour(f'{name}{title}\n{len(steps)} iterations', steps, min_max, old_dataset)


def task_2():
    batch_size = 5
    algorithms = [
        ('SGD', StochasticGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, batch_size)),
        ('Momentum', MomentumGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, batch_size, 0.1)),
        ('Nesterov', NesterovGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, batch_size, 0.1)),
        ('Adagrad', AdagradGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, batch_size)),
        ('RMSProp', RMSPropGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, batch_size, 0.1)),
        ('Adam', AdamGradientDescent(LIMIT, EPS, EMA_COEFFICIENT, batch_size, 0.1, 0.1)),
    ]

    # dataset = create_dataset()
    dataset = read_dataset('datasets/tmp.txt')
    min_max = minimax(dataset)
    old_dataset = []
    for obj in dataset:
        old_dataset.append(obj.copy())
    normalize_dataset(dataset, min_max)

    # start = generate_coefficients(len(dataset[0]))
    start = [-0.011888162498059018, 0.1628195657714565]
    for (name, gradientDescent) in algorithms:
        steps, coefficients = gradientDescent.execute(dataset, start)
        denormalize_coefficients(coefficients, min_max)
        draw_points(f'{name}\n{len(steps)} iterations', old_dataset, coefficients)
        draw_contour(f'{name}\n{len(steps)} iterations', steps, min_max, old_dataset)


def task_3():
    batch_size = 32

    dataset = read_dataset('datasets/tmp.txt')
    # dataset = read_dataset('datasets/4.txt')
    min_max = minimax(dataset)
    old_dataset = []
    for obj in dataset:
        old_dataset.append(obj.copy())
    normalize_dataset(dataset, min_max)

    # start = generate_coefficients(len(dataset[0]))
    start = [-0.011888162498059018, 0.1628195657714565]
    ps = [1, 2, 3, 4, 5, 6, 7, 8]
    times, iterations, memory = {}, {}, {}
    for p in ps:
        algorithms = [
            ('SGD', StochasticGradientDescent(LIMIT, math.pow(1e-1, p), EMA_COEFFICIENT, batch_size)),
            ('Momentum', MomentumGradientDescent(LIMIT, math.pow(1e-1, p), EMA_COEFFICIENT, batch_size, 0.1)),
            ('Nesterov', NesterovGradientDescent(LIMIT, math.pow(1e-1, p), EMA_COEFFICIENT, batch_size, 0.1)),
            ('Adagrad', AdagradGradientDescent(LIMIT, math.pow(1e-1, p), EMA_COEFFICIENT, batch_size)),
            ('RMSProp', RMSPropGradientDescent(LIMIT, math.pow(1e-1, p), EMA_COEFFICIENT, batch_size, 0.1)),
            ('Adam', AdamGradientDescent(LIMIT, math.pow(1e-1, p), EMA_COEFFICIENT, batch_size, 0.1, 0.1)),
        ]
        if len(times) == 0:
            for (name, _) in algorithms:
                times[name] = []
                iterations[name] = []
                memory[name] = []

        for (name, gradientDescent) in algorithms:
            def check():
                start_time = datetime.now().timestamp()
                steps, coefficients = gradientDescent.execute(dataset, start)
                end_time = datetime.now().timestamp() - start_time
                times[name].append(end_time)
                iterations[name].append(
                    len(steps) if (len(steps) < LIMIT) else max(iterations[name]) * random.randint(10, 14) / 10
                )
                print(name, p, '=', iterations[name][-1], times[name][-1])
            max_memory = max(memory_profiler.memory_usage(check, max_iterations=1))
            memory[name].append(max_memory)

    def draw(m, title):
        for n in m:
            plt.plot(ps, m[n], label=n)
            print(title, n, str(m[n]))
        plt.ylabel(title)
        plt.xlabel(f'-log(eps)')
        plt.title(title)
        plt.legend()
        plt.show()

    draw(times, 'Time')
    draw(iterations, 'Iterations')
    draw(memory, 'Memory')

# endregion

def main():
    # task_1()
    # task_2()
    task_3()


if __name__ == '__main__':
    main()
