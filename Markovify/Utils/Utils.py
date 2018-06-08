
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


def get_tags(input, flatten=False):
    """
    Get the tags from an input.

    :param input: list of tags, list of tuples (word, tag), list of lists of tags or list of lists of tuples (word, tag)
    :param flatten:
    :return: list of tags.
    """
    if type(input[0]) is str:  # The input is a list of tags.
        tag_list = input
    elif type(input[0]) is tuple:  # The input is a list of tuples (word, tag).
        tag_list = [tag for (word, tag) in input]
    else:  # The input is a list of lists.
        if type(input[0][0]) is str:  # The input is a list of lists of tags.
            tag_list = input
        elif type(input[0][0]) is tuple:  # The input is a list of lists of tuples (word, tag).
            tag_list = []
            for sentence in input:
                tag_list.append([tag for (word, tag) in sentence])

    if flatten:
        tag_list = [tag for sublist in tag_list for tag in sublist]
    return tag_list


def get_words(input, flatten=False):
    """
    Get the tags from an input.

    :param input: list of tags, list of tuples (word, tag), list of lists of tags or list of lists of tuples (word, tag)
    :param flatten:
    :return: list of tags.
    """
    if type(input[0]) is str:  # The input is a list of tags.
        word_list = input
    elif type(input[0]) is tuple:  # The input is a list of tuples (word, tag).
        word_list = [word for (word, tag) in input]
    else:  # The input is a list of lists.
        if type(input[0][0]) is str:  # The input is a list of lists of tags.
            word_list = input
        elif type(input[0][0]) is tuple:  # The input is a list of lists of tuples (word, tag).
            word_list = []
            for sentence in input:
                word_list.append([word for (word, tag) in sentence])

    if flatten:
        word_list = [word for sublist in word_list for word in sublist]
    return word_list
