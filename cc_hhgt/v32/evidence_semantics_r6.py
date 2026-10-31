"""Evidence semantics shared by refreshed and legacy-normalized event consumers.

A source row's assertion is not a calibrated probability. Unknown is not a
negative observation. Broad natural-language inference is deliberately absent.
"""
from __future__ import annotations
import math,re
from collections.abc import Mapping
from .evidence_semantics_r5 import cancer_context

UNKNOWN_WORDS={'','unknown','nan','none','na','n/a','null','unspecified'}
POSITIVE={'up','upregulated','up-regulated','upregulation','increase','increased','positive','activation','activated','activates','activating','enhanced','enhances','promotion','promotes','promoted'}
NEGATIVE={'down','downregulated','down-regulated','downregulation','decrease','decreased','negative','inhibition','inhibited','inhibits','inhibiting','repressed','represses','suppresses','suppressed'}
NEUTRAL={'0','neutral','no change','unchanged','no effect','no significant change'}

def clean(value):
    if value is None:return ''
    try:
        if isinstance(value,float) and not math.isfinite(value):return ''
    except TypeError:pass
    return str(value).strip()

def direction_target(value,sign=None):
    text=clean(value).lower()
    # A negative statement does not imply the opposite effect. Explicit neutral
    # phrases below are the only accepted neutral observations.
    if text in NEUTRAL:return 1
    if re.search(r'\b(no|not|without|neither|lack|lacks|lacking|absence|failed|fails)\b',text):return -1
    parsed=None
    if text in POSITIVE:parsed=2
    elif text in NEGATIVE:parsed=0
    elif text in {'1','+1','+'}:parsed=2
    elif text in {'-1','-'}:parsed=0
    elif text not in UNKNOWN_WORDS:return -1  # do not find "up" in "supported"
    s=clean(sign)
    numeric=None
    if s.lower() not in UNKNOWN_WORDS:
        try:
            v=float(s)
            if math.isfinite(v):numeric=2 if v>0 else 0 if v<0 else 1
        except ValueError:pass
    if parsed is not None and numeric is not None and parsed!=numeric:return -1
    return parsed if parsed is not None else numeric if numeric is not None else -1

def validate_context(row: Mapping,claimed):
    """A legacy PAN_CANCER label cannot override contradictory raw context."""
    answer=cancer_context(claimed)
    raw_fields=('disease_raw','disease','disease_name','disease_context','cancer_raw')
    observations=[clean(row.get(k)) for k in raw_fields if clean(row.get(k)).lower() not in UNKNOWN_WORDS]
    for raw in observations:
        resolved=cancer_context(raw)
        if resolved in {'CONTEXT_UNRESOLVED','CONTEXT_UNSPECIFIED'}:
            return 'CONTEXT_UNRESOLVED'  # retained in rejected/context report, never broadcast
        if answer=='PAN_CANCER' and resolved!='PAN_CANCER':answer=resolved
        elif resolved!='PAN_CANCER' and answer not in {'CONTEXT_UNRESOLVED','CONTEXT_UNSPECIFIED',resolved}:
            return 'CONTEXT_UNRESOLVED'
        elif answer in {'CONTEXT_UNRESOLVED','CONTEXT_UNSPECIFIED'}:answer=resolved
    return answer
