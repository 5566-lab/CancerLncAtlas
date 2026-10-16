"""Tiny raw segment materialization verifies source hashes, never trains."""
from argparse import Namespace
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.current_core_export import sha256
from scripts import materialize_v32_directional_cnv as directional


@pytest.mark.parametrize("change_during_read",[False,True])
@pytest.mark.parametrize("threshold",[.3,.7])
def test_directional_materializer_binds_actual_raw_segments(tmp_path,monkeypatch,change_during_read,threshold):
    assert socket.gethostname()=="149"
    old=tmp_path/"old";partition=old/"cancer=BRCA";partition.mkdir(parents=True)
    def write(p,d):p.write_text(json.dumps(d))
    write(old/"SUCCESS.json",{"cancers":{"BRCA":{"path":str(partition)}}})
    write(partition/"SUCCESS.json",{"status":"SUCCESS"})
    write(partition/"patient_ids.json",["TCGA-AB-1234"])
    write(partition/"lncrna_ids.json",["L1"]);write(partition/"pathway_ids.json",["P1"])
    for entity in ("lncrna","pathway"):np.save(partition/f"{entity}_callable.npy",np.ones((1,1),np.uint8))
    membership=tmp_path/"membership.parquet"
    pd.DataFrame({"pathway_id":["P1"],"gene_id":["G1"]}).to_parquet(membership,index=False)
    intervals=tmp_path/"intervals.parquet"
    pd.DataFrame({"entity_id":["L1","G1"],"chromosome":["1","1"],"start":[10,30],"end":[20,40]}).to_parquet(intervals,index=False)
    staging=tmp_path/"staging";source=staging/"BRCA"/"TCGA-AB-1234.segments.txt";source.parent.mkdir(parents=True)
    pd.DataFrame({"Chromosome":["1"],"Start":[1],"End":[100],"Segment_Mean":[.5]}).to_csv(source,sep="\t",index=False)
    args=Namespace(cancer="BRCA",old_root=old,staging_root=staging,membership=membership,
                   intervals=intervals,output_root=tmp_path/"output",event_threshold=threshold)
    Path(args.output_root).mkdir()
    if change_during_read:
        original=pd.read_csv
        def changed(p,*a,**kw):
            frame=original(p,*a,**kw)
            if Path(p)==source:source.write_text(source.read_text()+"\n")
            return frame
        monkeypatch.setattr(pd,"read_csv",changed)
        with pytest.raises(RuntimeError,match="changed while reading"):directional.materialize_cancer(args)
        assert not (args.output_root/"cancer=BRCA").exists()
    else:
        success=directional.materialize_cancer(args)
        output=args.output_root/"cancer=BRCA";sources=output/"SOURCE_FILES.json"
        assert success["files"]["SOURCE_FILES.json"]==sha256(sources)
        assert json.loads(sources.read_text())["records"][0]["sha256"]==sha256(source)
        assert success["old_success_path"]==str(partition/"SUCCESS.json")
        assert np.load(output/"lncrna_signed_mean.npy")[0,0]==pytest.approx(.5)
        assert success["event_threshold"]==threshold
        assert bool(np.load(output/"lncrna_amplification.npy")[0,0])==(.5>=threshold)
        assert bool(np.load(output/"pathway_amplification.npy")[0,0])==(.5>=threshold)


def test_event_materializer_rejects_raw_drift(tmp_path,monkeypatch):
    assert socket.gethostname()=="149"
    from test_v32_segment_cnv_streaming import _fixture
    from cc_hhgt.v32 import segment_cnv_streaming as event
    candidates,folds,intervals,membership,staging,selected=_fixture(tmp_path)
    original=pd.read_csv
    def changed(p,*a,**kw):
        frame=original(p,*a,**kw);path=Path(p);path.write_text(path.read_text()+"\n");return frame
    monkeypatch.setattr(pd,"read_csv",changed)
    with pytest.raises(event.StreamingCNVError,match="changed while reading"):
        event.materialize_cancer_partition(cancer_id="BRCA",staging_root=staging,output_root=tmp_path/"output",
            candidates=candidates,folds=folds,intervals=intervals,membership=membership,selected_rows=selected,
            context_sha256="a"*64,max_memory_bytes=2_000_000_000)
    assert not (tmp_path/"output/cancer=BRCA").exists()
