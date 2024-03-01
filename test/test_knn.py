import pytest

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from src.model_selection import train_test_split
from src.utils import shuffle

from src.KNN import KNeighborsClassifier


def get_data():
    iris = datasets.load_iris()

    return iris["data"], iris["target"]


def test_knn():
    X, y = get_data()
    X, y = shuffle(X, y)

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    ss = StandardScaler().fit(X_train)
    X_train, X_test = ss.transform(X_train), ss.transform(X_test)

    knn = KNeighborsClassifier(k=5)

    knn.fit(X_train, y_train)
    acc = knn.evaluate(X_test, y_test)

    assert acc > 0.9
