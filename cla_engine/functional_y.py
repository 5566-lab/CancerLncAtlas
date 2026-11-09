"""Read cancer-specific functional relationship targets on fixed identity axes."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np

ARRAY_NAMES = ('membership', 'direction', 'observed_assay_bits',
               'strict_assay_bits', 'supported_assay_bits')
ASSAY_BITS = {'RNA': 1, 'PROTEIN': 2, 'PHOSPHO': 4, 'DNA_METHYLATION': 8,
              'DNA_COPY_NUMBER': 16, 'DNA_VARIANT': 32, 'OTHER': 64}


@dataclass(frozen=True)
class FunctionalRelationContext:
    cancer_id: str
    context: str
    lncrna_ids: tuple[str, ...]
    canonical_ids: tuple[str, ...]
    source_patients: tuple[str, ...]
    label_patients: tuple[str, ...]
    arrays: dict[str, np.ndarray]

    def counts(self):
        membership = self.arrays['membership']
        observed = self.arrays['observed_assay_bits'] != 0
        return {
            'cancer_id': self.cancer_id, 'context': self.context,
            'lncrnas': len(self.lncrna_ids), 'canonical_targets': len(self.canonical_ids),
            'observable_relations': int(observed.sum()),
            'strong_positive': int((membership == 2).sum()),
            'weak_positive': int((membership == 1).sum()),
            'unlabeled_observed': int(((membership == 0) & observed).sum()),
            'direction_discordant': int((self.arrays['direction'] == 3).sum()),
        }


def validate_relation_arrays(arrays, shape):
    if set(arrays) != set(ARRAY_NAMES):
        raise ValueError('Functional Y arrays do not match the declared schema')
    for name, value in arrays.items():
        if value.dtype != np.uint8 or value.shape != shape:
            raise ValueError(f'Functional Y shape or dtype differs: {name}')
    membership, direction = arrays['membership'], arrays['direction']
    observed, strict, supported = (arrays[k] for k in ARRAY_NAMES[2:])
    if np.any(membership > 2) or np.any(direction > 3) or np.any(observed > 127):
        raise ValueError('Unknown membership, direction or assay code')
    if np.any(supported & ~observed) or np.any(strict & ~supported):
        raise ValueError('Supported assay must be observed; strict must be supported')
    bit_count = np.array([int(v).bit_count() for v in range(256)], np.uint8)
    expected = np.where(bit_count[strict] >= 2, 2, np.where(supported != 0, 1, 0))
    if not np.array_equal(membership, expected):
        raise ValueError('Membership differs from distinct assay-family support')
    if np.any((membership == 0) != (direction == 0)):
        raise ValueError('Direction requires supported association evidence')


def _unique(values, name):
    if not all(isinstance(v, str) and v for v in values) or len(set(values)) != len(values):
        raise ValueError(f'Nonunique or invalid identities: {name}')
    return tuple(values)


def load_functional_context(catalog, cancer_id, context):
    """Load and verify one completed context; membership zero means unlabelled.

    Each assay-family bit is counted once, irrespective of how many algorithms
    provide its readouts. Patient scores remain in their source-specific units.
    """
    catalog = Path(catalog).resolve()
    base = (catalog / cancer_id / context).resolve()
    if not base.is_relative_to(catalog) or base == catalog:
        raise ValueError('Context must be inside the specified catalog')
    receipt = json.loads((base / 'COMPLETE.json').read_text())
    if receipt.get('cancer_id') != cancer_id or receipt.get('context') != context:
        raise ValueError('Context receipt identity differs')
    for name in ['AXES.json', 'SPLIT.json', 'Y.npz']:
        digest = hashlib.sha256((base / name).read_bytes()).hexdigest()
        if digest != receipt['files'][name]:
            raise ValueError(f'Context file checksum differs: {name}')
    axes = json.loads((base / 'AXES.json').read_text())
    split = json.loads((base / 'SPLIT.json').read_text())
    lncs = _unique(axes['lncrna_ids'], 'lncRNAs')
    targets = _unique(axes['canonical_ids'], 'canonical targets')
    source = _unique(split['source_patients'], 'source patients')
    label = _unique(split['label_patients'], 'label patients')
    if set(source) & set(label):
        raise ValueError('Source and label patients overlap')
    with np.load(base / 'Y.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    validate_relation_arrays(arrays, (len(lncs), len(targets)))
    result = FunctionalRelationContext(cancer_id, context, lncs, targets, source, label, arrays)
    for key, value in result.counts().items():
        if key in receipt and receipt[key] != value:
            raise ValueError(f'Context count differs: {key}')
    return result


def functional_task_weights(roles):
    """Pathway=1.0, State=0.5; a shared identity has one Pathway-primary task."""
    result = []
    for value in roles:
        parts = set(value.split('|'))
        if not parts or parts - {'Pathway', 'State'}:
            raise ValueError(f'Unknown functional target role: {value}')
        result.append(1.0 if 'Pathway' in parts else 0.5)
    return np.asarray(result, dtype=np.float32)
