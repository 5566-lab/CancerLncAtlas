"""Dense R6.2 rank associations grouped by actual observation masks.

All Cartesian pairs are evaluated. No expression/correlation admission filter.
The local raw signed CNV adjustment acts on the lncRNA residual only (A1).
"""
import numpy as np
from scipy.stats import rankdata,t as student_t

def groups(mask):
    _,inv=np.unique(np.packbits(mask.T,axis=1),axis=0,return_inverse=True)
    return [np.flatnonzero(inv==g) for g in range(int(inv.max())+1)] if len(inv) else []

def basis(z):
    u,s,_=np.linalg.svd(z,full_matrices=False)
    tol=max(z.shape)*np.finfo(float).eps*(s[0] if len(s) else 1)
    q=int((s>tol).sum())
    return u[:,:q],q

def bh(p):
    shape=p.shape;v=np.asarray(p,float).ravel();q=np.full(v.shape,np.nan)
    ix=np.flatnonzero(np.isfinite(v));order=ix[np.argsort(v[ix],kind='stable')]
    if len(order):q[order]=np.minimum(1,np.minimum.accumulate((v[order]*len(order)/np.arange(1,len(order)+1))[::-1])[::-1])
    return q.reshape(shape)

def associate(X,Y,C,call,Z,cohort,min_n=12,compute_fdr=True):
    X,Y,C,Z=map(lambda a:np.asarray(a,float),(X,Y,C,Z));call=np.asarray(call,bool);cohort=np.asarray(cohort,int)
    if X.shape!=C.shape or X.shape!=call.shape or X.shape[0]!=Y.shape[0] or X.shape[0]!=Z.shape[0]:raise ValueError('Observation axes differ')
    if len(set(cohort))!=len(cohort):raise ValueError('Repeated cohort patient')
    x,y,c,z=X[cohort],Y[cohort],C[cohort],Z[cohort]
    xm=np.isfinite(x)&np.isfinite(c)&call[cohort]&np.isfinite(z).all(1)[:,None];ym=np.isfinite(y)
    r=np.full((x.shape[1],y.shape[1]),np.nan);p=r.copy();nn=np.zeros(r.shape,np.int32);ranks=np.zeros(r.shape,np.int16)
    xgroups=groups(xm);ygroups=groups(ym);qcache={};nblocks=0
    for xi in xgroups:
      for yi in ygroups:
        valid=xm[:,xi[0]]&ym[:,yi[0]];n=int(valid.sum());dest=np.ix_(xi,yi);nn[dest]=n
        if n<min_n:continue
        key=np.packbits(valid).tobytes()
        if key not in qcache:qcache[key]=basis(z[valid])
        Q,q=qcache[key];xx=rankdata(x[np.ix_(valid,xi)],axis=0);yy=rankdata(y[np.ix_(valid,yi)],axis=0);cc=c[np.ix_(valid,xi)].copy()
        xx-=Q@(Q.T@xx);yy-=Q@(Q.T@yy);cc-=Q@(Q.T@cc)
        den=np.sum(cc*cc,axis=0);tol=1e-18*np.maximum(1,np.sum(c[np.ix_(valid,xi)]**2,axis=0));active=den>tol
        beta=np.divide(np.sum(cc*xx,axis=0),den,out=np.zeros(len(xi)),where=active);xr=xx-cc*beta
        rr=q+active.astype(int);df=n-rr-1;ranks[dest]=np.broadcast_to(rr[:,None],(len(xi),len(yi)))
        norm=np.sqrt(np.sum(xr*xr,axis=0)[:,None]*np.sum(yy*yy,axis=0)[None,:]);good=(norm>1e-10)&(df[:,None]>0)
        corr=np.divide(xr.T@yy,norm,out=np.full(norm.shape,np.nan),where=good);corr=np.clip(corr,-1,1)
        stat=np.abs(corr)*np.sqrt(np.maximum(df[:,None],0)/np.maximum(1-corr*corr,1e-15))
        r[dest]=corr;p[dest]=np.where(good,2*student_t.sf(stat,np.maximum(df[:,None],1)),np.nan);nblocks+=1
    out={'r':r,'p':p,'n':nn,'rank':ranks,'available':np.isfinite(r)&np.isfinite(p),'mask_groups':{'x':len(xgroups),'y':len(ygroups),'evaluated_blocks':nblocks}}
    if compute_fdr:out['q']=bh(p)
    return out
