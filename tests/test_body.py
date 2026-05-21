import numpy as np
from xenobot_alife.body import BodyGenome


def test_connected_body():
    b = BodyGenome(np.array([[1, 1, 0], [0, 1, 2]], dtype=np.int8))
    assert b.is_connected()


def test_disconnected_body():
    b = BodyGenome(np.array([[1, 0, 0], [0, 0, 2]], dtype=np.int8))
    assert not b.is_connected()
