"""
febrl_math.py: File for different math calculations used in febrl

"""

import math


def distL1(vec1: list, vec2: list) -> float:

    """
    L1 distance measure, also called Manhattan distance.

    The distance between two points measured along axes at right angles.

    See also:
        http://www.nist.gov/dads/HTML/lmdistance.html
        http://en.wikipedia.org/wiki/Distance

    Assumes the vectors are the same length.

    :param vec1: A list representing a point in len(vec1) dimensions
    :param vec2: A list representing a point in len(vec2) dimensions
    :return: The L1 Distance between vec1 and vec2
    """

    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors of different lengths are not supported. "
            f"Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}"
        )

    vec_len = len(vec1)

    l1_distance = 0.0

    for i in range(vec_len):
        l1_distance += abs(float(vec1[i]) - float(vec2[i]))

    return l1_distance


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def distL2(vec1: list, vec2: list) -> float:

    """
    L2 distance measure, also known as the Euclidean distance.

    See also:
        http://www.nist.gov/dads/HTML/lmdistance.html
        http://en.wikipedia.org/wiki/Distance

    Assumes the vectors are the same length.

    :param vec1: A list representing a point in len(vec1) dimensions
    :param vec2: A list representing a point in len(vec2) dimensions
    :return: The L2 Distance between vec1 and vec2
    """

    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors of different lengths are not supported. "
            f"Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}"
        )

    vec_len = len(vec1)

    L2_distance = 0.0

    for i in range(vec_len):
        x = float(vec1[i]) - float(vec2[i])
        L2_distance += x * x

    return math.sqrt(L2_distance)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def distLInf(vec1, vec2):

    """
    L-Infinity distance measure. The maximum absolute difference between
    two vector components.

    See also:
        http://www.nist.gov/dads/HTML/lmdistance.html
        http://en.wikipedia.org/wiki/Distance
    :param vec1: A list representing a point in len(vec1) dimensions
    :param vec2: A list representing a point in len(vec2) dimensions
    :return: The L2 Distance between vec1 and vec2
    """

    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors of different lengths are not supported. "
            f"Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}"
        )

    vec_len = len(vec1)

    Linf_distance = 0.0

    for i in range(vec_len):
        x = abs(float(vec1[i]) - float(vec2[i]))
        Linf_distance = max(x, Linf_distance)

    return Linf_distance


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def distCanberra(vec1, vec2):
    """Canberra distance measure.

       See also:
       http://people.revoledu.com/kardi/tutorial/Similarity/CanberraDistance.html
    """

    #  assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    cbr_dist = 0.0

    for i in range(vec_len):
        x = abs(float(vec1[i]) - float(vec2[i]))
        y = abs(float(vec1[i])) + abs(float(vec2[i]))
        if y > 0.0:
            cbr_dist += x / y

    return cbr_dist


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculates and returns the cosine similarity between two vectors

    See also:
        https://en.wikipedia.org/wiki/Cosine_similarity

    :param vec1: A list representing a vector of len(vec1) dimensions
    :param vec2: A list representing a vector of len(vec2) dimensions
    :return: The cosine similarity.
    """

    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors of different lengths are not supported. "
            f"Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}"
        )

    combine_vector = zip(vec1, vec2)

    vec1sqr = 0.0
    vec2sqr = 0.0
    dot_product = 0.0

    for val in combine_vector:
        vec1sqr += val[0] * val[0]
        vec2sqr += val[1] * val[1]
        dot_product += val[0] * val[1]

    if vec1sqr == 0.0 or vec2sqr == 0.0:
        return 0.0  # At least one vector is all zeros

    else:
        vec1_magnitude = math.sqrt(vec1sqr)
        vec2_magnitude = math.sqrt(vec2sqr)

        cos_sim = dot_product / (vec1_magnitude * vec2_magnitude)

        # Due to rounding errors the similarity can be slightly larger than 1.0
        cos_sim = min(cos_sim, 1.0)
        cos_sim = max(cos_sim, -1.0)

        return cos_sim


def dist_cosine(vec1, vec2):

    """
    Cosine distance is 1 - the Cosine Similarity, A measure of similarity between two positive vectors.
    - How similarly oriented are two vectors?

    Note: This function assumes that all vector elements are non-negative.

    See also:
            http://en.wikipedia.org/wiki/Vector_space_model
    :param vec1: A list representing a point in len(vec1) dimensions
    :param vec2: A list representing a point in len(vec2) dimensions
    :return: The L2 Distance between vec1 and vec2
    """

    if len(vec1) != len(vec2):
        raise ValueError(
            f"Vectors of different lengths are not supported. "
            f"Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}"
        )

    cos_sim = cosine_similarity(vec1, vec2)

    if cos_sim < 0:
        raise ValueError(
            f"Only positive vectors are supported for dist_cosine: "
            f"cosine_similarity: {cos_sim}, vec1: {vec1}, vec2: {vec2}"
        )

    cos_dist = 1.0 - cos_sim

    return cos_dist


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## TODO, PC Jan 2008 ***********


def distMahalanobis(vec1, vec2):
    """Mahalanobis distance measure.

       See also:
         http://en.wikipedia.org/wiki/Mahalanobis_distance
    """

    assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    mal_dist = 0.0

    return mal_dist
