from febrl import distance
import pytest


def test_distL1():
    """
    - distL1 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0
    - For a vectors <0, 0> and <0, 5>, result should be 5
    - For a vectors <-3, -3> and <3, 3>, result should be 12

    :return:
    """

    assert distance.distL1([0, 0], [0, 0]) == 0
    assert distance.distL1([0, 0], [0, 5]) == 5
    assert distance.distL1([-3, -3], [3, 3]) == 12

    with pytest.raises(ValueError) as e_info:
        distance.distL1([0, 0], [0, 0, 0])


def test_distL2():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0
    - For a vectors <0, 0> and <0, 5>, result should be 5
    - For a vectors <-3, -3> and (3, 3>, result should be 8.4853

    :return:
    """

    assert distance.distL2([0, 0], [0, 0]) == 0
    assert distance.distL2([0, 0], [0, 5]) == 5
    assert (
        round(distance.distL2([-3, -3], [3, 3]), 4) == 8.4853
    )  # 8.4852813742385702928101323452582

    with pytest.raises(ValueError) as e_info:
        distance.distL2([0, 0], [0, 0, 0])


def test_distLInf():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors <0, 0> and <0, 0>, result should be 0 => |0 - 0|
    - For a vectors <0, 0> and <0, 5>, result should be 5 => |5 - 1|
    - For a vectors <-3, -3> and <3, 3>, result should be 6 => |3 - (-3)|
    - For a vectors <0, -3> and <3, 0>, result should be 3 => |0 - (-3)|

    :return:
    """

    assert distance.distLInf([0, 0], [0, 0]) == 0
    assert distance.distLInf([0, 0], [0, 5]) == 5
    assert distance.distLInf([-3, -3], [3, 3]) == 6
    assert distance.distLInf([0, -3], [3, 0]) == 3

    with pytest.raises(ValueError) as e_info:
        distance.distLInf([0, 0], [0, 0, 0])
