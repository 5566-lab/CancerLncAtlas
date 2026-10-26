"""Compact, private patient scopes captured from actual native statistics.

Aligned candidate pairs and graph Cartesian products reference shared patient
row sets. This avoids expanding patients x candidates or all gene edges.
The scopes describe statistical computation, not learned-parameter updates.
"""
import json

import numpy as np
import pandas as pd

from .contracts import canonical_json_sha256
from .local_cnv_confounding import canonical_patient

FORMAT = 'NATIVE_LOCAL_CNV_STATISTICAL_PATIENT_SCOPES_V1'
TABLE = 'local_cnv_patient_scopes_private'
COLUMNS = ['model_id','dependency_context','patient_fold_id','source_role','source_release_sha256',
    'output_sha256','patient_map_sha256','cancer_id','statistic','split_role','cohort_id',
    'patient_map_row_indices_json','native_patient_count','pairing','lncrna_ids_json','target_ids_json',
    'target_type','candidate_pair_count','calculation_entered','finite_statistic_json','format']


def encoded(value):
    return json.dumps(value, separators=(',',':'), ensure_ascii=False)


class Trace:
    def __init__(self, patient_path, patient_pin, cancer, fold):
        self.patient_pin = patient_pin; self.cancer = cancer; self.fold = fold
        self.patients = pd.read_csv(patient_path, sep='\t', dtype=str)
        self.canonical = self.patients.patient_id.map(canonical_patient)
        self.cancer_rows = self.patients.cancer_id.eq(cancer)
        self.events = []

    def emit(self, statistic, split, patients, lncrnas, targets, target_type,
             pairing, entered, finite=None):
        patients = list(map(str,patients)); lncrnas = list(map(str,lncrnas)); targets = list(map(str,targets))
        if len(patients) != len(set(patients)):
            raise ValueError('Statistical patient scope repeats a native patient')
        mask = self.cancer_rows & self.canonical.isin(patients)
        if set(self.canonical.loc[mask]) != set(patients):
            raise ValueError('Statistical patient scope is absent from its fold authority')
        actual_folds = set(pd.to_numeric(self.patients.loc[mask,'patient_fold_id'],errors='raise'))
        allowed = set(range(5))-{self.fold,(self.fold+1)%5} if split=='train' else {(self.fold+1)%5}
        if not actual_folds <= allowed:
            raise ValueError('Statistical patient scope crosses its declared split')
        if pairing=='ALIGNED_PAIRS' and len(lncrnas)!=len(targets):
            raise ValueError('Statistical patient scope has unaligned pairs')
        if finite is not None and (pairing!='ALIGNED_PAIRS' or len(finite)!=len(lncrnas)):
            raise ValueError('Statistical patient scope has unaligned finite flags')
        rows = np.flatnonzero(mask.to_numpy()).tolist()
        cohort = canonical_json_sha256({'patient_map_sha256':self.patient_pin['sha256'],'rows':rows})
        self.events.append({'patient_map_sha256':self.patient_pin['sha256'],'cancer_id':self.cancer,
            'statistic':statistic,'split_role':split,'cohort_id':cohort,'patient_map_row_indices_json':encoded(rows),
            'native_patient_count':len(patients),'pairing':pairing,'lncrna_ids_json':encoded(lncrnas),
            'target_ids_json':encoded(targets),'target_type':target_type,
            'candidate_pair_count':len(lncrnas) if pairing=='ALIGNED_PAIRS' else len(lncrnas)*len(targets),
            'calculation_entered':bool(entered),
            'finite_statistic_json':encoded(None if finite is None else np.asarray(finite,dtype=bool).tolist()),
            'format':FORMAT})

    def full(self, split):
        def observe(pairs, patients, entered, finite):
            self.emit('A0_FULL',split,patients,pairs.lncrna_id,pairs.pathway_id,
                      'pathway','ALIGNED_PAIRS',entered,finite)
        return observe

    def absent(self, pairs, split):
        for statistic in ('A0_MATCHED_A1_LOCAL','A1_ON_A2_A2_LOCAL_PATHWAY'):
            self.emit(statistic,split,[],pairs.lncrna_id,pairs.pathway_id,
                      'pathway','ALIGNED_PAIRS',False,np.zeros(len(pairs),dtype=bool))

    def matched(self, pairs, patients, split, *, only_a1=False):
        pairs = pairs.reset_index(drop=True); patients = np.asarray(patients,dtype=str)
        def observe(local, pathway, result, minimum):
            choices = [('A0_MATCHED_A1_LOCAL',local,result['available'])]
            if not only_a1: choices.append(('A1_ON_A2_A2_LOCAL_PATHWAY',pathway,result['available_a2']))
            for statistic, mask, available in choices:
                if mask.shape != (len(patients),len(pairs)):
                    raise ValueError('Statistical patient scope mask alignment differs')
                _, inverse = np.unique(np.packbits(mask,axis=0).T,axis=0,return_inverse=True)
                for group in range(int(inverse.max())+1 if len(inverse) else 0):
                    columns = np.flatnonzero(inverse==group); rows = mask[:,columns[0]]
                    chosen = pairs.iloc[columns]
                    self.emit(statistic,split,patients[rows],chosen.lncrna_id,chosen.pathway_id,
                              'pathway','ALIGNED_PAIRS',int(rows.sum())>=minimum,available[columns])
        return observe

    def graph(self, statistic, patients, lncrnas, genes, entered):
        self.emit(statistic,'train',patients,lncrnas,genes,'gene','CARTESIAN',entered)

    def payload(self):
        return {'patient_map':self.patient_pin,'events':self.events}


def attach(inventory, payload, binding, root, model, role, fold):
    """Map actual fold-authority rows, keeping pair identities and masks private."""
    pin = payload['patient_map']
    path = inventory.bind(pin,root,model,role+'.statistical_patient_authority',fold)
    patients = pd.read_csv(path,sep='\t',dtype=str)
    seen = set()
    for event in payload['events']:
        row = {'model_id':model,'dependency_context':inventory.context_id,'patient_fold_id':fold,
            'source_role':role,'source_release_sha256':binding['sources']['sha256'],
            'output_sha256':binding['output']['sha256'],**event}
        inventory.statistical_patient_scopes.append(row)
        phase = 'MODEL_SELECTION' if event['split_role']=='validation' else 'FEATURE_CONSTRUCTION'
        key = (event['cohort_id'],event['split_role'],event['calculation_entered'])
        if key in seen: continue
        seen.add(key)
        indices = json.loads(event['patient_map_row_indices_json'])
        scope = ('NATIVE_STATISTICAL_INPUT_PATIENT_ROWS' if event['calculation_entered'] else
                 'NATIVE_STATISTICAL_ELIGIBILITY_PATIENT_ROWS')
        inventory.patient_use.observe([patients.iloc[indices][['patient_id']]],digest=pin['sha256'],
            model=model,context=inventory.context_id,role=role+'.cohort.'+event['cohort_id'],fold=fold,
            phase=phase,scope=scope)
