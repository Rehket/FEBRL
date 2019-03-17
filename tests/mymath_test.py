
from febrl import mymath
import pytest

def test_distL1():
    """
    - distL1 should raise a Value Exception if lengths are not the same.
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
