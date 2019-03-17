
"""
File for different distance calculations

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
        raise ValueError(f'Vectors of different lengths are not supported. '
                         f'Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}')

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
        raise ValueError(f'Vectors of different lengths are not supported. '
                         f'Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}')

    vec_len = len(vec1)

    L2_distance = 0.0

    for i in range(vec_len):
        x = float(vec1[i]) - float(vec2[i])
        L2_distance += x * x

    return math.sqrt(L2_distance)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def distLInf(vec1, vec2):

    """
    L-Infinity distance measure.

    See also:
        http://www.nist.gov/dads/HTML/lmdistance.html
        http://en.wikipedia.org/wiki/Distance
    :param vec1: A list representing a point in len(vec1) dimensions
    :param vec2: A list representing a point in len(vec2) dimensions
    :return: The L2 Distance between vec1 and vec2
    """

    if len(vec1) != len(vec2):
        raise ValueError(f'Vectors of different lengths are not supported. '
                         f'Length of vec1: {len(vec1)}, Length of vec2: {len(vec2)}')

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


def distCosine(vec1, vec2):
    """Cosine distance measure.

       Note: This function assumes that all vector elements are non-negative.

       See also:
         http://en.wikipedia.org/wiki/Vector_space_model
    """

    assert len(vec1) == len(vec2)

    vec_len = len(vec1)

    vec1sum = 0.0
    vec2sum = 0.0
    vec12sum = 0.0

    for i in range(vec_len):
        vec1sum += vec1[i] * vec1[i]
        vec2sum += vec2[i] * vec2[i]
        vec12sum += vec1[i] * vec2[i]

    if vec1sum * vec2sum == 0.0:
        cos_dist = 1.0  # At least one vector is all zeros

    else:
        vec1sum = math.sqrt(vec1sum)
        vec2sum = math.sqrt(vec2sum)

        cos_sim = vec12sum / (vec1sum * vec2sum)

        # Due to rounding errors the similarity can be slightly larger than 1.0
        #
        cos_sim = min(cos_sim, 1.0)

        assert (cos_sim >= 0.0) and (cos_sim <= 1.0), (cos_sim, vec1, vec2)

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
