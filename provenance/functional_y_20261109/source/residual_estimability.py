import numpy as np
from scipy.stats import rankdata
from block_associations import groups,basis

def residual_estimability(X,Y,C,call,Z,cohort,previous_available,min_n=12,previous_n=None,previous_rank=None):
    X,Y,C,Z=map(lambda a:np.asarray(a,float),(X,Y,C,Z));call=np.asarray(call,bool);cohort=np.asarray(cohort,int)
    x,y,c,z=X[cohort],Y[cohort],C[cohort],Z[cohort];old=np.asarray(previous_available,bool)
    assert old.shape==(x.shape[1],y.shape[1]) and x.shape==c.shape
    xm=np.isfinite(x)&np.isfinite(c)&call[cohort]&np.isfinite(z).all(axis=1)[:,None];ym=np.isfinite(y)
    codes=np.zeros(old.shape,np.uint8);cache={};checked=0
    for xi in groups(xm):
      for yi in groups(ym):
        dest=np.ix_(xi,yi);before=old[dest]
        if not before.any():continue
        valid=xm[:,xi[0]]&ym[:,yi[0]];n=int(valid.sum());assert valid.dtype==bool
        if n<min_n:
          codes[dest]=np.where(before,1,0);continue
        key=np.packbits(valid).tobytes()
        if key not in cache:cache[key]=basis(z[valid])
        Q,q=cache[key];xx=rankdata(x[np.ix_(valid,xi)],axis=0);yy=rankdata(y[np.ix_(valid,yi)],axis=0);cc=c[np.ix_(valid,xi)].copy()
        xvar=np.ptp(xx,axis=0)>0;yvar=np.ptp(yy,axis=0)>0;raw_xnorm=np.linalg.norm(xx,axis=0);raw_ynorm=np.linalg.norm(yy,axis=0)
        xr=xx-Q@(Q.T@xx);yr=yy-Q@(Q.T@yy);cr=cc-Q@(Q.T@cc)
        den=np.sum(cr*cr,axis=0);active=den>1e-18*np.maximum(1,np.sum(cc*cc,axis=0));beta=np.divide(np.sum(cr*xr,axis=0),den,out=np.zeros(len(xi)),where=active);xr-=cr*beta
        rr=q+active.astype(int);df=n-rr-1
        if previous_n is not None:assert np.all(np.asarray(previous_n)[dest][before]==n),'Pair patient counts changed'
        if previous_rank is not None:
          expected=np.broadcast_to(rr[:,None],before.shape);assert np.array_equal(np.asarray(previous_rank)[dest][before],expected[before]),'Covariate ranks changed'
        # Rank inputs have a natural scale. This rejects numerical null vectors,
        # including constant expression and exact adjustment-span relationships.
        factor=256*np.finfo(float).eps*max(n,z.shape[1]+1)
        xok=np.linalg.norm(xr,axis=0)>factor*np.maximum(1,raw_xnorm);yok=np.linalg.norm(yr,axis=0)>factor*np.maximum(1,raw_ynorm)
        cause=np.zeros(before.shape,np.uint8)
        cause|=np.broadcast_to((df<=0).astype(np.uint8)[:,None]*2,before.shape)
        cause|=np.broadcast_to((~xvar).astype(np.uint8)[:,None]*4,before.shape)
        cause|=np.broadcast_to((~yvar).astype(np.uint8)[None,:]*8,before.shape)
        cause|=np.broadcast_to((~xok).astype(np.uint8)[:,None]*16,before.shape)
        cause|=np.broadcast_to((~yok).astype(np.uint8)[None,:]*32,before.shape)
        codes[dest]=np.where(before,cause,0);checked+=int(before.sum())
    return old&(codes==0),codes,dict(previous_available=int(old.sum()),retained_available=int((old&(codes==0)).sum()),removed=int((codes>0).sum()),checked=checked,raw_constant_x=int(((codes&4)>0).sum()),raw_constant_y=int(((codes&8)>0).sum()),residual_null_x=int(((codes&16)>0).sum()),residual_null_y=int(((codes&32)>0).sum()),tolerance='256 * machine_epsilon * max(n, design_columns+1) * max(1, input_rank_vector_norm)')
