
def increment(dictionary, k1, k2):
    """
    dictionary[k1][k2]++

    :param dictionary: Dictionary of dictionary of integers.
    :param k1: First key.
    :param k2: Second key.
    :return: same dictionary with incremented [k1][k2]
    """
    if k1 not in dictionary:
        dictionary[k1] = {}
    if 0 not in dictionary[k1]:
        dictionary[k1][0] = 0
    if k2 not in dictionary[k1]:
        dictionary[k1][k2] = 0

    dictionary[k1][0] += 1   # k1 count
    dictionary[k1][k2] += 1  # k1, k2 pair count

    return dictionary
