# @Author  : Han Yue <yuehan1523@gmail.com>
# @License : Apache License 2.0
from fairpy.dataset import Adult
import numpy as np
import math


def test_Adult():
    dataset = Adult(download=True)
    X, y, s = dataset.get_X_y_s()
    assert len(dataset.df) == len(X)
    assert len(dataset.df) == len(y)
    assert len(dataset.df) == 45222
    assert len(np.unique(y)) > 1
    assert np.any(y) >= 0

    feat_idx = dataset.feat_idx
    assert feat_idx != None

    split_data = dataset.split(val_p=0.2, test_p=0.2)
    assert math.ceil(len(dataset.df)*0.2) == len(split_data.X_val)
    assert math.ceil(len(dataset.df)*0.2) == len(split_data.X_test)
    assert len(split_data.X_train) == len(split_data.y_train)