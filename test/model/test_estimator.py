from my_check_estimator import check_estimator
from fairpy.model import LabelBias, LinearFairERM, FairCstr, DIRemover, EqOddsCalib, FairGLM, IFair, FairRank
import pytest

@pytest.mark.parametrize('model', [LabelBias, LinearFairERM, FairCstr, EqOddsCalib, FairGLM, IFair])
def test_estimator_checks(model):
    estimator = model()
    check_estimator(estimator)

@pytest.mark.parametrize('model', [DIRemover])
def test_estimator_checks_DIRemover(model):
    estimator = model(s_idx=0)
    check_estimator(estimator)

@pytest.mark.parametrize('model', [FairRank])
def test_estimator_checks_FairRank(model):
    estimator = model(K=20, P=0.5, alpha=0.10)
    check_estimator(estimator)