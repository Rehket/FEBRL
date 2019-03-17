from febrl import febrl_math
import pytest


def test_distL1():
    """
    - distL1 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0
    - For a vectors <0, 0> and <0, 5>, result should be 5
    - For a vectors <-3, -3> and <3, 3>, result should be 12

    :return:
    """

    assert febrl_math.distL1([0, 0], [0, 0]) == 0
    assert febrl_math.distL1([0, 0], [0, 5]) == 5
    assert febrl_math.distL1([-3, -3], [3, 3]) == 12

    with pytest.raises(ValueError) as e_info:
        febrl_math.distL1([0, 0], [0, 0, 0])


def test_distL2():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0
    - For a vectors <0, 0> and <0, 5>, result should be 5
    - For a vectors <-3, -3> and (3, 3>, result should be 8.4853

    :return:
    """

    assert febrl_math.distL2([0, 0], [0, 0]) == 0
    assert febrl_math.distL2([0, 0], [0, 5]) == 5
    assert (
        round(febrl_math.distL2([-3, -3], [3, 3]), 4) == 8.4853
    )  # 8.4852813742385702928101323452582

    with pytest.raises(ValueError) as e_info:
        febrl_math.distL2([0, 0], [0, 0, 0])


def test_distLInf():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0 => |0 - 0|
    - For a vectors <0, 0> and <0, 5>, result should be 5 => |5 - 1|
    - For a vectors <-3, -3> and <3, 3>, result should be 6 => |3 - (-3)|
    - For a vectors <0, -3> and <3, 0>, result should be 3 => |0 - (-3)|

    :return:
    """

    assert febrl_math.distLInf([0, 0], [0, 0]) == 0
    assert febrl_math.distLInf([0, 0], [0, 5]) == 5
    assert febrl_math.distLInf([-3, -3], [3, 3]) == 6
    assert febrl_math.distLInf([0, -3], [3, 0]) == 3

    with pytest.raises(ValueError) as e_info:
        febrl_math.distLInf([0, 0], [0, 0, 0])


def test_cosine_similarity():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0 => |0 - 0|
    - For a vectors <0, 0> and <0, 5>, result should be 5 => |5 - 1|
    - For a vectors <-3, -3> and <3, 3>, result Raise value exception
    - For a vectors <0, -3> and <3, 0>, result Raise value exception

    :return:
    """

    assert febrl_math.cosine_similarity([0, 0], [0, 0]) == 0  # Zero Vector
    assert febrl_math.cosine_similarity([1, 0], [0, 5]) == 0  # Orthogonal Vectors
    assert (
        round(febrl_math.cosine_similarity([1, 3], [2, 5]), 5) == 0.99827
    )  # 0.99827437317499593042850507243421
    assert febrl_math.cosine_similarity([-3, -3], [3, 3]) == -1  # Opposite Vectors
    assert febrl_math.cosine_similarity([0, -3], [3, 0]) == 0  # Orthogonal

    with pytest.raises(ValueError) as e_info:
        febrl_math.cosine_similarity([0, 0], [0, 0, 0])


def test_distCosine():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0 => |0 - 0|
    - For a vectors <0, 0> and <0, 5>, result should be 5 => |5 - 1|
    - For a vectors <-3, -3> and <3, 3>, result Raise value exception
    - For a vectors <0, -3> and <3, 0>, result Raise value exception

    :return:
    """

    assert febrl_math.dist_cosine([0, 0], [0, 0]) == 1
    assert febrl_math.dist_cosine([1, 0], [0, 5]) == 1
    assert round(febrl_math.dist_cosine([1, 3], [2, 5]), 5) == 0.00173

    with pytest.raises(ValueError) as e_info:
        febrl_math.dist_cosine([-3, -3], [3, 3])
        febrl_math.dist_cosine([0, -3], [3, 0])
        febrl_math.dist_cosine([0, 0], [0, 0, 0])
