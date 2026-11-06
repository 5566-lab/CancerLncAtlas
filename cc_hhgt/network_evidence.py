"""Evidence gate for a physical RNA-protein interaction prior.

An experimental abundance/perturbation association is not binding evidence.
Never infer physical binding from the word 'experimental' or a relation label.
"""
PHYSICAL_CLASSES=frozenset(['eclip','other_clip','rip','rna_capture','other_physical'])
PHYSICAL_ASSAYS=frozenset(['eclip','clip_unspecified','hits_clip','iclip','par_clip','rip','chart','chirp','rap','rna_pulldown','co_ip','emsa'])

def physical_rpi_mask(frame):
    required={'graph_assay_class','experiment_family','assay_subtype'}
    if not required.issubset(frame.columns):raise ValueError('RPI_METHOD_EVIDENCE_REQUIRED')
    explicit=frame.assay_subtype.fillna('').astype(str).map(lambda v:bool(v) and set(v.split('|')).issubset(PHYSICAL_ASSAYS))
    return frame.graph_assay_class.isin(PHYSICAL_CLASSES)&frame.experiment_family.eq('physical_binding')&explicit


def require_physical_network_receipt(audit):
    if not str(audit.get('rpi_policy','')).startswith('EXPLICIT_PHYSICAL_METHOD_AND_FAMILY_REQUIRED'):
        raise ValueError('RPI_EVIDENCE_GATE_MISSING: historical inputs admitted abundance/reporter/unspecified associations; regenerate or materialize verified physical-network inputs before training')
