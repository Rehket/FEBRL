
from febrl import mymath
import pytest

def test_distL1():
    """
    - distL1 should raise a Value Exception if vector lengths are not the same.
    - For a vectors (0, 0) and (0, 0), result should be 0
    - For a vectors (0, 0) and (0, 5), result should be 5
    - For a vectors (-3, -3) and (3, 3), result should be 12

    :return:
    """

    assert mymath.distL1([0, 0], [0, 0]) == 0
    assert mymath.distL1([0, 0], [0, 5]) == 5
    assert mymath.distL1([-3, -3], [3, 3]) == 12

    with pytest.raises(ValueError) as e_info:
        mymath.distL1([0, 0], [0, 0, 0])


def test_distL2():
    """
    - distL2 should raise a Value Exception if vector lengths are not the same.
    - For a vectors (0, 0) and (0, 0), result should be 0
    - For a vectors (0, 0) and (0, 5), result should be 5
    - For a vectors (-3, -3) and (3, 3), result should be 8.4853

    :return:
    """

    assert mymath.distL2([0, 0], [0, 0]) == 0
    assert mymath.distL2([0, 0], [0, 5]) == 5
    assert round(mymath.distL2([-3, -3], [3, 3]), 4) == 8.4853  # 8.4852813742385702928101323452582

    with pytest.raises(ValueError) as e_info:
        mymath.distL2([0, 0], [0, 0, 0])
