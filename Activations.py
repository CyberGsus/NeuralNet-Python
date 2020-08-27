from math import tanh

E = 2.71828182845904523536


def LogisticFunction(x: float):
    return 1 / (1 + pow(E, -x))


def LogisticFunctionDerivative(x: float):
    return LogisticFunction(x) * (1 - LogisticFunction(x))


def TanH(x: float):
    return tanh(x)


def TanHDerivative(x: float):
    return 1 - TanH(x) * TanH(X)


def ReLU(x: float):
    if x < 0:
        return 0
    return x


def ReLUDerivative(x: float):
    if x < 0:
        return 0
    return 1
