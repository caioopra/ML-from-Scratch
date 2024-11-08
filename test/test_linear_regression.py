import pytest
import math

from src.Value import Value
from src.models import LinearRegression


def create_data(WEIGHT, BIAS) -> tuple[list, list, list, list]:
    start = 0
    end = 1
    step = 0.02

    X = [start + i * step for i in range(int((end - start) / step))]
    y = [WEIGHT * x + BIAS for x in X]

    train_split = int(0.8 * len(X))

    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    return X_train, y_train, X_test, y_test


def test_linear_regression():
    lr = LinearRegression()

    WEIGHT = 0.7
    BIAS = 0.3

    LR = 0.01
    EPOCHS = 1000

    X_train, y_train, X_test, y_test = create_data(WEIGHT, BIAS)

    for k in range(EPOCHS):
        y_pred = [lr([x]) for x in X_train]
        loss = sum(
            ((yout - ygt) ** 2 for ygt, yout in zip(y_train, y_pred)), Value(0.0)
        )

        for p in lr.parameters():
            p.grad = 0

        loss.backward()

        for p in lr.parameters():
            p.data += -LR * p.grad

    print("Linear regression parameters: ", lr.parameters())

    assert math.isclose(lr.parameters()[0].data, WEIGHT)
    assert math.isclose(lr.parameters()[1].data, BIAS)
