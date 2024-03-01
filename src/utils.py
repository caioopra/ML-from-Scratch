from random import shuffle as rd_shuffle

def shuffle(*arrays):
    """Given a list of arrays, shuffles them in the same order.

    Args:
        *arrays (list): list of arrays.

    Returns:
        list: list of the arrays shuffled in the same order.
    """
    paired = list(zip(*arrays))

    rd_shuffle(paired)

    return list(zip(*paired))
