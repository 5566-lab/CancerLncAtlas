import numpy as np
import pytest
from cla_engine.relation_graph_view import validate_graph_arrays
from test_joint_node_omics import _model_fixture


def test_endpoint_and_missing_value_masks_are_enforced():
    spec,_,arrays=_model_fixture(True)
    names={k:['f'+str(i) for i in range(6)] for k in spec['nodes']}
    assert len(validate_graph_arrays(spec['nodes'],arrays,names))==len(spec['relations'])
    key=next(k for k in arrays if k.startswith('edge::'))
    arrays[key][0,0]=999
    with pytest.raises(ValueError,match='EDGE_ENDPOINT'):
        validate_graph_arrays(spec['nodes'],arrays,names)
    arrays[key][0,0]=0
    arrays['node_omics::gene'][1,2]=.7
    with pytest.raises(ValueError,match='OMICS_MASK'):
        validate_graph_arrays(spec['nodes'],arrays,names)
