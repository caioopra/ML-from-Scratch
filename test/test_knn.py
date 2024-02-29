import pytest

from ucimlrepo import fetch_ucirepo 

from src.KNN import KNeighborsClassifier

def get_data():
    data = iris = fetch_ucirepo(id=53) 

    X = data.data.features
    y = data.data.targets
    
    print(X, y)


def test_knn():
    accs = []

    ks = range(1, 30)    
    
    for k in ks:
        knn = KNeighborsClassifier(k=k)
        
        knn.fit(X_train)