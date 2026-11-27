"""Replace RNA measurement versions without selecting biological target identities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def replace_coding_readouts(values, metadata, coding_values, coding_metadata):
    """Match original score IDs exactly; preserve other assays and source readouts.

    An unreviewed external RNA definition is retained as a source observation,
    with an explicit lineage field. It is not asserted to use coding-only inputs.
    """
    if (not values.index.is_unique or not values.columns.is_unique
            or not coding_values.index.is_unique or not coding_values.columns.is_unique
            or not metadata.target_id.is_unique or not coding_metadata.source_score_id.is_unique
            or not coding_metadata.readout_id.is_unique):
        raise ValueError('CODING_REPLACEMENT_DUPLICATE_AXIS')
    if set(values.index) != set(coding_values.index):
        raise ValueError('CODING_REPLACEMENT_PATIENT_MISMATCH')
    if set(values.columns) != set(metadata.target_id):
        raise ValueError('CODING_REPLACEMENT_SOURCE_AXIS')
    md=metadata.set_index('target_id').loc[values.columns].rename_axis('target_id').reset_index()
    lookup=coding_metadata.set_index('source_score_id')
    result=values.copy()
    mapped=[]
    for i,row in md.iterrows():
        old=row.target_id
        is_rna=str(row.measurement_layer).startswith('RNA')
        item={'original_target_id':old,'target_id':old,'original_column_index':row.column_index,
              'column_index':i,'coding_expression_basis_verified':False,
              'measurement_definition':'PRESERVED_NON_RNA_SOURCE',
              'applied_method':'PRESERVED_SOURCE_READOUT','method_variant':'SOURCE_VERSION'}
        if is_rna and old in lookup.index:
            definition=lookup.loc[old]
            new=definition.readout_id
            if new not in coding_values:
                raise ValueError('CODING_REPLACEMENT_MISSING_SCORES')
            result[old]=coding_values[new].reindex(values.index)
            item.update(target_id=new,coding_expression_basis_verified=True,
                        measurement_definition='RECOMPUTED_PROTEIN_CODING_GENE_ID_INPUT',
                        applied_method=definition.method,method_variant=definition.method_variant)
        elif is_rna:
            item['measurement_definition']='PRESERVED_EXTERNAL_RNA_SOURCE'
        mapped.append(item)
    mapping=pd.DataFrame(mapped)
    result.columns=mapping.target_id.tolist()
    if not result.columns.is_unique:
        raise ValueError('CODING_REPLACEMENT_OUTPUT_COLLISION')
    if np.isinf(result.to_numpy(float)).any():
        raise ValueError('CODING_REPLACEMENT_INFINITE_VALUE')
    for key in mapping:
        md[key]=mapping[key].to_numpy()
    md['patients_in_scope']=len(result)
    md['patients_measured']=[int(result[c].notna().sum()) for c in result]
    md['measured_fraction']=md.patients_measured/len(result)
    md['nonconstant']=[result[c].nunique(dropna=True)>1 for c in result]
    return result,md,mapping
