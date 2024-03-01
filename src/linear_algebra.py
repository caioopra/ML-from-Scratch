from math import sqrt


def manhattan_distance(v1, v2):
    """Return the manhattan distance between two vectors."""
    if len(v1) != len(v2):
        raise "Vectors must have the same length."

    return sum([abs(x1 - x2) for x1, x2 in zip(v1, v2)])


def euclidian_distance(v1, v2):
    """Return the euclidian distance between two vectors."""
    if len(v1) != len(v2):
        raise "Vectors must have the same length."

    return sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(v1, v2)]))
