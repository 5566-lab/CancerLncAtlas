"""Samplewise signatures on a fixed, gene-ID-qualified coding background."""
from __future__ import annotations
import pandas as pd
from cc_hhgt.m0_joint_scores import raw_ssgsea


def score_fixed_coding_set(expression, annotation, members, name):
    if not annotation.gene_id.is_unique or not annotation.gene_biotype.eq('protein_coding').all():
        raise ValueError('FIXED_SIGNATURE_REQUIRES_CODING_GENE_IDS')
    symbols=set(annotation.reference_symbol.dropna())- {''}
    if set(expression.columns)!=symbols or not expression.columns.is_unique or not expression.index.is_unique:
        raise ValueError('FIXED_SIGNATURE_BACKGROUND_AXIS')
    members=list(members)
    if not members or len(members)!=len(set(members)) or not set(members)<=symbols:
        raise ValueError('FIXED_SIGNATURE_MEMBER_DEFINITION')
    scores,coverage=raw_ssgsea(expression,{name:members},integer_ranks=True)
    return scores,coverage
