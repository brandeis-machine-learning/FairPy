# @Author  : Han Yue <yuehan1523@gmail.com>
# @License : Apache License 2.0
from fairpy.dataset import Adult, Bank, Compas, Credit, Dutch, German, Oulad
import numpy as np
import math
import pytest




@pytest.mark.parametrize("dataset", [Adult, Bank, Compas, Credit, Dutch, German, Oulad])
def test_dataset(dataset):
    # normal test
    data = dataset(download=True)
    X, y, s = data.get_X_y_s()
    assert len(data.df) == len(X)
    assert len(data.df) == len(y)
    assert len(np.unique(y)) > 1
    assert np.any(y) >= 0

    feat_idx = data.feat_idx
    assert feat_idx != None

    split_data = data.split(val_p=0.1, test_p=0.2)
    assert math.ceil(len(data.df)*0.1) == len(split_data.X_val)
    assert math.ceil(len(data.df)*0.2) == len(split_data.X_test)
    assert len(split_data.X_train) == len(split_data.y_train)


def test_dataset_split():
    data = Adult(download=True)
    with pytest.raises(ValueError) as exceptInfo:
        split_data = data.split(val_p=0.0, test_p=0.0)
    excepted_msg=exceptInfo.value.args[0]
    assert len(excepted_msg) > 1

    with pytest.raises(ValueError) as exceptInfo:
        split_data = data.split(val_p=0.6, test_p=0.6)
    excepted_msg=exceptInfo.value.args[0]
    assert excepted_msg[:10] == 'test_size='




# @pytest.mark.parametrize("dataset", [Adult, Bank, Compas, Credit, Dutch, German, Oulad])
def test_download():
    with pytest.raises(FileExistsError) as exceptInfo:
        data = Adult(download=False, dir_path='wahaha')
    excepted_msg=exceptInfo.value.args[0]
    assert excepted_msg[-31:]=='consider set `download` to True'




@pytest.mark.parametrize("sen_feat", ['sex', 'wahaha', ['sex', 'race'], ['sex', 'race', 'wahaha'], Adult])
def test_sen_feat(sen_feat):
    try:
        data = Adult(download=True, sen_feat=sen_feat)
    except KeyError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1