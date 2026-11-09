from pathlib import Path
import socket,json,hashlib,numpy as np
from scipy.stats import rankdata
from block_associations import associate
from residual_estimability import residual_estimability
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;rng=np.random.default_rng(20261109);n=120;X=rng.normal(size=(n,8));Y=rng.normal(size=(n,5));C=rng.normal(size=X.shape);Z=np.column_stack([np.ones(n),rng.normal(size=(n,3))]);call=np.ones(X.shape,bool);idx=np.arange(n)
a=associate(X,Y,C,call,Z,idx);v,code,report=residual_estimability(X,Y,C,call,Z,idx,a['available'],previous_n=a['n'],previous_rank=a['rank']);assert v.all() and not code.any()
Xs=X.copy();Xs[:,0]=0;Ys=Y.copy();Ys[:,0]=7;v,code,_=residual_estimability(Xs,Ys,C,call,Z,idx,np.ones((8,5),bool));assert not v[0,:].any() and not v[:,0].any();assert v[1:,1:].all()
# A tiny expression scale retains ranks and must not become an expression filter.
v2,_,_=residual_estimability(X*1e-100,Y,C,call,Z,idx,np.ones((8,5),bool));assert v2.all()
# A rank vector in the covariate span has no residual association estimand.
zc=np.column_stack([Z,rankdata(X[:,1])]);v3,code,_=residual_estimability(X,Y,C,call,zc,idx,np.ones((8,5),bool));assert not v3[1,:].any();assert ((code[1,:]&16)>0).all()
# Local A1 CNV can also explain the entire residual rank vector.
cc=C.copy();cc[:,2]=rankdata(X[:,2]);v4,code,_=residual_estimability(X,Y,cc,call,Z,idx,np.ones((8,5),bool));assert not v4[2,:].any();assert ((code[2,:]&16)>0).all()
# Observation masks are boolean even when the source callable file is uint8.
call[:5,3]=False;v5,_,_=residual_estimability(X,Y,C,call.astype(np.uint8),Z,idx,np.ones((8,5),bool));assert v5.all()
perm=np.array([7,3,0,5,6,1,4,2]);vp,_,_=residual_estimability(X[:,perm],Y,C[:,perm],call[:,perm],Z,idx,np.ones((8,5),bool));assert np.array_equal(vp,v5[perm,:])
checks=['RANDOM_NONDEGENERATE_PAIRS_UNCHANGED','RAW_CONSTANT_X_AND_Y_UNDEFINED','TINY_EXPRESSION_SCALE_RETAINED','COVARIATE_SPAN_NULL_UNDEFINED','A1_CNV_SPAN_NULL_UNDEFINED','CALLABLE_UINT8_CAST_TO_BOOL','FEATURE_PERMUTATION_INVARIANT']
rec={'target_host':'149','status':'PASS','checks':checks,'implementation_sha256':hashlib.sha256((R/'residual_estimability.py').read_bytes()).hexdigest()};(R/'RESIDUAL_ESTIMABILITY_VALIDATION.json').write_text(json.dumps(rec,indent=2));print(rec)
