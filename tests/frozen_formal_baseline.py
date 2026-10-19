"""Explicit zero-coefficient L1 serialization fixture; no fit or optimizer."""
from types import SimpleNamespace
import numpy as np
from cc_hhgt.v32.baselines import build_safe_hashed_design


def frozen_l1(frames, **kwargs):
    model = SimpleNamespace(coef_=np.zeros((1,4096)), intercept_=np.zeros(1), classes_=np.array([0,1]),
        decision_function=lambda x: np.zeros(x.shape[0]))
    return model, {"synthetic_fixed_no_fit": True}, *(build_safe_hashed_design(frames[k], n_features=4096) for k in ("train","validation"))
