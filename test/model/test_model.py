# @Author  : Han Yue <yuehan1523@gmail.com>
# @License : Apache License 2.0
from sklearn.preprocessing import StandardScaler
from fairpy.model import LabelBias, LinearFairERM
from fairpy.dataset import Adult
import numpy as np
import pytest




@pytest.mark.parametrize('model', [LabelBias, LinearFairERM])
def test_model(model):
    dataset = Adult(sen_feat="sex", download=True)
    split_data = dataset.split(val_p=0.75, test_p=0.2)
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(split_data.X_train)
    X_test_scale = scaler.fit_transform(split_data.X_test)

    cur_model = model()
    cur_model.fit(X_train_scale, split_data.y_train, split_data.s_train)
    y_pred = cur_model.predict(X_test_scale)
    assert len(y_pred) == len(split_data.y_test)




def test_fit():
    X_train = np.random.random([50,20])
    y_train = np.random.randint(2, size=50)
    s_train = np.random.randint(10, size=50)
    cur_model = LabelBias()

    with pytest.raises(ValueError) as exceptInfo:
        cur_model.fit(X_train, y_train[:-1], s_train)
    excepted_msg=exceptInfo.value.args[0]
    assert excepted_msg[:50] == 'Found input variables with inconsistent numbers of'

    with pytest.raises(ValueError) as exceptInfo:
        cur_model.fit(X_train, y_train, np.array([0] + [1,1,1]))
    excepted_msg=exceptInfo.value.args[0]
    assert len(excepted_msg) > 0

    with pytest.raises(ValueError) as exceptInfo:
        cur_model.fit(X_train, y_train, np.zeros(100))
    excepted_msg=exceptInfo.value.args[0]
    assert len(excepted_msg) > 0