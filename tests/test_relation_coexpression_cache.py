import numpy as np
from cla_engine.graph import coexpression


def test_cached_gene_residuals_preserve_pairwise_missingness_edges_and_support():
    rng=np.random.default_rng(51);n=48
    x=rng.normal(size=(n,19));g=rng.normal(size=(n,23));cnv=rng.normal(size=x.shape)
    z=np.column_stack([np.ones(n),rng.normal(size=n)])
    x[::7,::3]=np.nan;g[::9,::5]=np.nan;cnv[::11,::4]=np.nan
    g[:,0]=x[:,0]+rng.normal(scale=.05,size=n)
    options=dict(topk=4,min_n=12,return_audit=True,lnc_batch=3,gene_batch=7)
    old,old_audit=coexpression(x,g,cnv,np.isfinite(cnv),z,np.arange(n),**options)
    for limit in (0,1024**2):
        new,audit=coexpression(x,g,cnv,np.isfinite(cnv),z,np.arange(n),cache_gene_residuals=True,gene_cache_max_bytes=limit,**options)
        for sign in old:
            np.testing.assert_array_equal(new[sign][0],old[sign][0])
            np.testing.assert_allclose(new[sign][1],old[sign][1],rtol=1e-6,atol=1e-7)
            for key in ('n','rank'):np.testing.assert_array_equal(audit['edge_support'][sign][key],old_audit['edge_support'][sign][key])
        for key in ('estimable_pairs','pairs_below_minimum_n','rank_or_variance_degenerate_pairs','selected_edges'):
            assert audit[key]==old_audit[key]
