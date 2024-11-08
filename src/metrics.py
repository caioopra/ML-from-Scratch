from math import sqrt

from typing import List

from src.Value import Value


def mae(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates mean absolute error, given two lists of values"""
    mae_val = sum(
        (abs(y - y_hat) for y, y_hat in zip(y_true, y_pred)), Value(0.0)
    ) / len(y_true)

    return Value(mae_val)


def mse(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates mean squared error, given two lists of values"""
    mse_val = sum(
        ((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)), Value(0.0)
    ) / len(y_true)

    return Value(mse_val)


def rmse(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates root mean squared error, given two lists of values"""
    rmse_val = (
        sum(((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)), Value(0.0))
        / len(y_true)
    ) ** (0.5)

    return Value(rmse_val)


def sse(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates sum of squared errors, given two lists of values"""
    return sum(((yout - ygt) ** 2 for ygt, yout in zip(y_true, y_pred)), Value(0.0))
