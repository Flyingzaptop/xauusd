import numpy as np
import pytest
from scripts import score_market_v31 as score
from src.inference_v31 import binary_metrics


def test_calibration_cannot_load_labels_before_all_fits(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(score,'load_split',lambda *a:pytest.fail('Labels loaded before barrier'))
    with pytest.raises(RuntimeError,match='All15'):
        score.calibrate({'models':['a'],'seeds':[7],'output':'absent'},{})


def test_evaluation_cannot_load_labels_before_calibrator_freeze(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(score,'read_fits',lambda *a:[])
    monkeypatch.setattr(score,'load_split',lambda *a:pytest.fail('Evaluation loaded before freeze'))
    with pytest.raises(FileNotFoundError):
        score.evaluate({'output':'absent'}, {})


def test_stable_logit_loss_and_invariant_auc_when_sigmoid_saturates():
    z=np.array([-80.,-79.,79.,80.]); y=np.array([0.,1.,0.,1.])
    raw=binary_metrics(y,z)
    scaled=binary_metrics(y,100*z+20,ranking_logits=z)
    assert raw['auc']==scaled['auc']==0.75
    assert np.isfinite(scaled['log_loss'])
    assert scaled['log_loss']>1000
    assert sum(r['count'] for r in scaled['calibration'])==4
