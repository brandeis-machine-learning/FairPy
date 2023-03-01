# @Author  : Han Yue <yuehan1523@gmail.com>
# @License : Apache License 2.0
from sklearn.linear_model import LinearRegression
from fairpy.model import LinearFairERM
# from fairpy.model import FairERM
import numpy as np
import pytest


@pytest.mark.parametrize('estimator', ['wahaha', 100, None, LinearRegression])
def test_estimator(estimator):
    X_train = np.random.random([50,20])
    y_train = np.random.randint(2, size=50)
    s_train = np.random.randint(10, size=50)

    try:
        cur_model = LinearFairERM(estimator=estimator)
        cur_model.fit(X_train, y_train, s_train)
    except ValueError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1
    except TypeError as exceptInfo:
        assert len(exceptInfo.args[0]) > 1




# def my_kernel(x1, x2):
#     return np.dot(x1, np.transpose(x2))

# @pytest.mark.parametrize('kernel', ['rbf', 'linear', 'wahaha', 100, my_kernel])
# def test_metric(kernel):
#     X_train = np.random.random([50,20])
#     y_train = np.random.randint(2, size=50)
#     s_train = np.random.randint(10, size=50)

#     try:
#         cur_model = FairERM(kernel=kernel)
#         cur_model.fit(X_train, y_train, s_train)
#     except ValueError as exceptInfo:
#         assert len(exceptInfo.args[0]) > 1
#     except TypeError as exceptInfo:
#         assert len(exceptInfo.args[0]) > 1




# @pytest.mark.parametrize('value', [100, 'wahaha', 6.5, 0, -10])
# def test_other_input(value):
#     X_train = np.random.random([50,20])
#     y_train = np.random.randint(2, size=50)
#     s_train = np.random.randint(10, size=50)

#     try:
#         cur_model = FairERM(C=value)
#         cur_model.fit(X_train, y_train, s_train)
#     except ValueError as exceptInfo:
#         assert len(exceptInfo.args[0]) > 1
#     except TypeError as exceptInfo:
#         assert len(exceptInfo.args[0]) > 1

#     try:
#         cur_model = FairERM(gamma=value)
#         cur_model.fit(X_train, y_train, s_train)
#     except ValueError as exceptInfo:
#         assert len(exceptInfo.args[0]) > 1
#     except TypeError as exceptInfo:
#         assert len(exceptInfo.args[0]) > 1