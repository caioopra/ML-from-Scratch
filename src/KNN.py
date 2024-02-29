from src.linear_algebra import euclidian_distance


class KNeighborsClassifier:
    def __init__(self, k=5, dist=euclidian_distance):
        self.k = k
        self.dist = dist

    def _most_commom(self, lst):
        return max(set(lst), key=lst.count)

    def _measure_distance(self, x1, X):
        """Meausres the distance between a point and a list of points

        Args:
            x1 (_type_): _description_
            X (_type_): _description_
        """
        return [self.dist(x1, x2) for x2 in X]

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test) -> list:
        neighbors = []

        for x in X_test:
            distances = self._measure_distance(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]

            neighbors.append(y_sorted[: self.k])

        return list(map(self._most_commom, neighbors))

    def evaluate(self, X_test, y_test):
        """
        Evaluates the accuracy of the model, given a set of test data.
        """
        y_pred = self.predict(X_test)

        acc = sum(
            filter(lambda y_true, y_pred: y_true == y_pred, zip(y_test, y_pred))
        ) / len(y_test)

        return acc
