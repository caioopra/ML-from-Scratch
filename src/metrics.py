from math import sqrt

from typing import List

from src.Value import Value


def mae(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates mean absolute error, given two lists of values"""
    mae_val = sum(abs(y - y_hat) for y, y_hat in zip(y_true, y_pred)) / len(y_true)

    return Value(mae_val)


def mse(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates mean squared error, given two lists of values"""
    mse_val = sum((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)) / len(y_true)

    return Value(mse_val)


def rmse(y_true: List[float], y_pred: List[float]) -> Value:
    """Calculates root mean squared error, given two lists of values"""
    rmse_val = sqrt(
        sum((y - y_hat) ** 2 for y, y_hat in zip(y_true, y_pred)) / len(y_true)
    )

    return Value(rmse_val)
