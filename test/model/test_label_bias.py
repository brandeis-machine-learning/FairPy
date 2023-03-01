# @Author  : Han Yue <yuehan1523@gmail.com>
# @License : Apache License 2.0
from sklearn.linear_model import LinearRegression
from fairpy.model import LabelBias
import numpy as np
import pytest




@pytest.mark.parametrize('metric', ['dp', 'eop', 'wahaha', 100])
def test_metric(metric):
    X_train = np.random.random([50,20])
    y_train = np.random.randint(2, size=50)
    s_train = np.random.randint(10, size=50)

    try:
        cur_model = LabelBias(metric=metric)
        cur_model.fit(X_train, y_train, s_train)
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1




@pytest.mark.parametrize('estimator', ['wahaha', 100, None, LinearRegression, LinearRegression()])
def test_estimator(estimator):
    X_train = np.random.random([50,20])
    y_train = np.random.randint(2, size=50)
    s_train = np.random.randint(10, size=50)

    try:
        cur_model = LabelBias(estimator=estimator)
        cur_model.fit(X_train, y_train, s_train)
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except AttributeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1




@pytest.mark.parametrize('value', [100, 'wahaha', None, 6.5, 0, -10])
def test_other_input(value):
    X_train = np.random.random([50,20])
    y_train = np.random.randint(2, size=50)
    s_train = np.random.randint(10, size=50)

    try:
        cur_model = LabelBias(max_iter=value)
        cur_model.fit(X_train, y_train, s_train)
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1

    try:
        cur_model = LabelBias(tol=value)
        cur_model.fit(X_train, y_train, s_train)
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1

    try:
        cur_model = LabelBias(lr=value)
        cur_model.fit(X_train, y_train, s_train)
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1