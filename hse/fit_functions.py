import numpy as np


def stepped_linear(x, a, b, c, d):
    return np.where(x < 8, a * x + b, c * x + d)


def linear(x, a, b):
    return a * x + b


def stepped_linear_2d(points, a, b, c, d, e, f):
    """
    Combination between a stepped linear function in the x dimension and a linear function in the y dimension.

    :param points: Input data, should have shape (n, 2), 1th column contains x data, 2nd column contains y data.
    :return: Returns output as list of shape (n,).
    """
    x, y = np.hsplit(points, 2)
    return (stepped_linear(x, a, b, c, d) * linear(y, e, f)).reshape(-1)
