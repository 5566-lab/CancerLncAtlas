"""Raw GMT/GTF -> original ATAC R/preparation/fusion inspection; never fit."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import native_annotation_workflow as reference
from cc_hhgt.v32 import native_pathway_membership_workflow as members
from cc_hhgt.v32 import atac_training as atac
from cc_hhgt.v32.atac_cohort import membership_reference, materialization_cohort
from cc_hhgt.v32.atac_materialization import run_atac_materialization, INPUT_CONTRACT_FORMAT
from cc_hhgt.v32.fold_local_scope import candidate_frame, CANDIDATES
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_atac_scope import materialized as legacy_atac, prepare, CANCERS, ROOT, no_fit
from test_v32_native_annotation_workflow import inputs as reference_input, gtf_text


@pytest.fixture(scope='module')
def native(legacy_atac,tmp_path_factory):
    root=tmp_path_factory.mktemp('native_atac_membership');_,old,_,_=legacy_atac
    genes=[f'ENSG{200000+i:011}' for i in range(123)]
    lines=gtf_text().splitlines()[:3]
    for i,gene in enumerate(genes):
        kind='lncRNA' if i<120 else 'protein_coding';symbol=f'L{i}' if i<120 else f'G{i-119}'
        lines.append(f'chr1\tHAVANA\tgene\t{i*100+1}\t{i*100+90}\t.\t+\t.\tgene_id "{gene}.1"; gene_type "{kind}"; gene_name "{symbol}";')
    # A second G3 gene lacks a measured promoter; do not collapse its ID into
    # the autosomal copy or imply that it was observed.
    lines.append(f'chrY\tHAVANA\tgene\t1\t90\t.\t+\t.\tgene_id "{genes[-1]}.1_PAR_Y"; gene_type "protein_coding"; gene_name "G3";')
    source=reference_input(root/'raw',text='\n'.join(lines)+'\n');ref=reference.run(source,root/'reference')
    gmt=root/'sets.gmt';gmt.write_text('P1\tsynthetic\tG1\tG2\tG3\tUNKNOWN\n')
    gm=root/'gmt.json';write(gm,{'format':'SSGSEA_GMT_INPUT_V1','collections':[{'collection':'HALLMARK','file':record(gmt)}]})
    request=root/'MEMBERS.json';write(request,{'format':members.INPUT,'reference_annotation':record(ref),'gmt_manifest':record(gm)})
    binding=members.run(request,root/'members');membership=Path(read(binding)['membership']['path'])
    definition={'format':CANDIDATES,'target_host':'149','training_started':False,'measured_detection_used':False,
        'annotation':read(ref)['annotation'],'gmt_manifest':record(gm),'cancers':CANCERS,'budget_per_cancer':120,'seed':20260726,
        'implementations':[record(ROOT/p) for p in ('cc_hhgt/v32/fold_local_scope.py','scripts/materialize_fold_local_scope.py','scripts/prepare_v32_formal.py')]}
    candidate=root/'candidates.parquet';candidate_frame(definition,root).to_parquet(candidate,index=False)
    definition['output']=record(candidate);write(root/'CANDIDATE_SOURCES.json',definition)
    promoters=root/'promoters.bed.gz';frame=pd.read_csv(old['promoters'],sep='\t');frame['gene_id']=genes
    frame.to_csv(promoters,sep='\t',index=False,compression='gzip')
    paths={**old,'candidates':candidate,'pathway_membership':membership,'promoters':promoters}
    contract=root/'contract.json';write_contract(contract,paths,record(root/'CANDIDATE_SOURCES.json'))
    from scripts.run_v32_atac_materialization import main
    values={'candidates':candidate,'patient-folds':paths['patient_folds'],'pathway-membership':membership,
        'matrix':paths['normalized_atac_matrix'],'mapping':paths['identifier_mapping'],'peak-set':paths['peak_set'],
        'promoters':promoters,'extractor-script':ROOT/'scripts/materialize_v32_atac_gene_accessibility.R',
        'input-contract':contract,'output-root':root/'materialized'}
    assert main([part for k,v in values.items() for part in ('--'+k,str(v))])==0
    success=root/'materialized/GENE_ACCESSIBILITY_SUCCESS.json'
    return root,paths,success,read(success)


def write_contract(path,paths,definition):
    value={'format':INPUT_CONTRACT_FORMAT,'contract_id':'SYNTHETIC_NATIVE_MEMBERSHIP',
        'files':[{**record(p),'bytes':p.stat().st_size,'basename':p.name,'role':role} for role,p in paths.items()]}
    if definition is not None:value['candidate_definition']=definition
    write(path,value)


def test_original_R_and_all_fold_features_use_actual_native_membership(native):
    root,paths,_,result=native;prepared=prepare(native)
    proof=result['membership_reference'];assert proof==read(root/'materialized/LAUNCH_PREFLIGHT.json')['membership_reference']
    assert proof['metadata']['unmapped_symbol_memberships']==1 and proof['metadata']['one_to_many_symbol_memberships']==1
    assert not proof['missing_member_imputed_zero'] and not proof['reference_authenticity_verified']
    required=pd.read_csv(root/'materialized/intermediate/required_gene_ids.tsv.gz',sep='\t').gene_id
    assert required.str.startswith('UNMAPPED_GMT:').sum()==1 and required.str.endswith('_PAR_Y').sum()==1
    assert len(prepared['membership'])==5
    accessibility=atac._load_accessibility(prepared['records']['BRCA'],'BRCA',prepared['folds'])
    candidates=atac._candidate_table(paths['candidates'],'BRCA')
    _,weights,counts,mapping=atac._pathway_projection(candidates,accessibility,prepared['membership'],3)
    assert counts.tolist()==[3] and set(mapping.structural_reason)=={''}
    expected=[accessibility.gene_index[f'ENSG{200000+i:011}'] for i in range(120,123)]
    np.testing.assert_allclose(weights.toarray()[0,expected],1/3)
    np.testing.assert_allclose(weights @ accessibility.values, accessibility.values[expected].mean(axis=0,keepdims=True),atol=1e-6)
    _,_,_,unavailable=atac._pathway_projection(candidates,accessibility,prepared['membership'],4)
    assert set(unavailable.structural_reason)=={'ATAC_EXACT_PATHWAY_PROMOTER_COVERAGE_LT_MINIMUM'}
    for fold,(features,score,audit) in atac.collect_atac_examples(prepared).items():
        assert features.shape==(120,7) and np.isfinite(features).all() and np.isfinite(score).all()
        assert audit[0]['sampled_pairs']==120
    assert result['raw_gap_cancers']==['OV']


def test_actual_auxiliary_and_fusion_preparation_inspect_reference_sources(native,tmp_path):
    from test_v32_native_auxiliary_workflow import test_actual_prepare_collect_cli_independent_audit_and_fusion as exercise
    exercise(('atac',native),tmp_path)
    from scripts.run_v32_current_multimodal_fusion import collect_inputs
    from cc_hhgt.v32.current_auxiliary_model_use import inspect_fusion_expert
    from cc_hhgt.v32.current_model_use import Inventory
    frame,audit,declarations,_=collect_inputs(tmp_path/'fusion.json',expected_core_run_id='current-G2',temp_directory=tmp_path/'inspect_tmp')
    inventory=Inventory([],tmp_path)
    inspect_fusion_expert(inventory,tmp_path/'prepared_binding.json',{'source_inputs':declarations},audit,frame,'atac','fusion','discovery')
    assert any(row['format']==members.FORMAT for row in inventory.transform_replays)
    launch=read(members.discover(native[1]['pathway_membership'])['preflight']['path'])
    paths={row['path'] for row in inventory.dependencies}
    assert launch['gtf']['path'] in paths and launch['gmt_sources']['HALLMARK']['path'] in paths
    member_roles={r['role'] for r in inventory.transform_replays if r['format']==members.FORMAT}
    assert not any(r['reason']=='RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED' and r['role'] in member_roles for r in inventory.gaps)
    assert any(r['reason']=='AUXILIARY_RAW_STUDY_AND_PATIENT_USAGE_NOT_FULLY_RECONSTRUCTED' for r in inventory.gaps)


@pytest.mark.parametrize('fault',['gmt','annotation','missing_definition'])
def test_materializer_rejects_mismatched_native_membership_before_R(native,tmp_path,monkeypatch,fault):
    root,paths,_,_=native;definition=read(root/'CANDIDATE_SOURCES.json')
    if fault=='gmt':
        gmt=tmp_path/'other.gmt';gmt.write_text('P1\tsynthetic\tG1\tG2\tG4\n')
        gm=tmp_path/'GMT.json';write(gm,{'format':'SSGSEA_GMT_INPUT_V1','collections':[{'collection':'HALLMARK','file':record(gmt)}]})
        definition['gmt_manifest']=record(gm)
    elif fault=='annotation':
        annotation=tmp_path/'annotation.tsv';annotation.write_bytes(Path(definition['annotation']['path']).read_bytes()+b'\n')
        definition['annotation']=record(annotation)
    definition_path=tmp_path/'CANDIDATE.json';write(definition_path,definition)
    contract=tmp_path/'INPUT.json';write_contract(contract,paths,None if fault=='missing_definition' else record(definition_path))
    import cc_hhgt.v32.atac_materialization as materializer
    def forbidden(*args,**kwargs):raise AssertionError('R must not launch for invalid membership')
    monkeypatch.setattr(materializer.subprocess,'run',forbidden)
    with pytest.raises(ValueError,match='different|requires'):
        run_atac_materialization(candidates_path=paths['candidates'],patient_folds_path=paths['patient_folds'],
            pathway_membership_path=paths['pathway_membership'],matrix_path=paths['normalized_atac_matrix'],
            mapping_path=paths['identifier_mapping'],peak_set_path=paths['peak_set'],promoters_path=paths['promoters'],
            extractor_script_path=ROOT/'scripts/materialize_v32_atac_gene_accessibility.R',input_contract_path=contract,
            output_root=tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('fault',['missing_reference','changed_reference','removed_scope'])
def test_training_preparation_rejects_rehashed_downgraded_member_proof(native,tmp_path,fault):
    root,paths,_,result=native;value=deepcopy(result)
    if fault=='missing_reference':value.pop('membership_reference')
    elif fault=='changed_reference':value['membership_reference']['metadata']['unmapped_symbol_memberships']=0
    else:
        value.pop('cohort');value.pop('input_contract_path');value.pop('membership_reference')
    path=tmp_path/'SUCCESS.json';write(path,value)
    with pytest.raises(ValueError,match='membership'):
        atac.prepare_atac_inputs(candidates_path=paths['candidates'],patient_folds_path=paths['patient_folds'],
            pathway_membership_path=paths['pathway_membership'],materialization_success_path=path)
