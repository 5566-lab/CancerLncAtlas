"""Exercise the explicit code-only verification gate before parameter updates."""
import os
import pytest


@pytest.mark.skipif(os.environ.get('CANCERLNCATLAS_CODE_ONLY_ATAC_CHECKS')!='1',reason='Opt-in code-only verification launcher')
@pytest.mark.parametrize('entry',['atac','cnv','genomic','fusion','crossfit','Adam','AdamW','SGD'])
def test_code_only_session_rejects_parameter_fitting(entry):
    import torch
    from cc_hhgt.v32 import atac_training,cnv_only_training_v2,genomic_training,multimodal_fusion
    functions={'atac':atac_training._fit_logistic,'cnv':cnv_only_training_v2._fit_head,
        'genomic':genomic_training._fit_head,'fusion':multimodal_fusion.fit_residual_fusion_model,
        'crossfit':multimodal_fusion.train_pair_blocked_crossfit_fusion,
        **{name:getattr(torch.optim,name) for name in ('Adam','AdamW','SGD')}}
    with pytest.raises(AssertionError,match='Code-only ATAC verification forbids parameter fitting'):
        functions[entry]()
