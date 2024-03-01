# TODO: accept any number of arrays and accept shuffling
def train_test_split(
    X: list, y: list, test_size: int = None, train_size: int = None
) -> tuple[list]:
    """Split arrays or matrices into random train and test subsets"""
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if test_size is not None and train_size is not None:
        if test_size + train_size != 1:
            raise ValueError("test_size + train_size must be equal to 1")
    elif test_size is not None:
        train_size = 1 - test_size
    elif train_size is not None:
        test_size = 1 - train_size
    else:
        raise ValueError("Either test_size or train_size must be provided")

    if not 0 < test_size < 1 or not 0 < train_size < 1:
        raise ValueError("test_size and train_size must be between 0 and 1")

    index = int(len(X) * train_size)
    X_train, y_train = X[:index], y[:index]
    X_test, y_test = X[index:], y[index:]

    return X_train, y_train, X_test, y_test
