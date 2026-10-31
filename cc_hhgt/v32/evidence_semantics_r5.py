"""Conservative event semantics; never turn prediction/unknown context into experiment/pan-cancer."""
from __future__ import annotations
import re
CANCERS=set('ACC BLCA BRCA CESC CHOL COAD DLBC ESCA GBM HNSC KICH KIRC KIRP LAML LGG LIHC LUAD LUSC MESO OV PAAD PCPG PRAD READ SARC SKCM STAD TGCT THCA THYM UCEC UCS UVM'.split())
ALIASES={
 'breast cancer':'BRCA','breast carcinoma':'BRCA','breast invasive carcinoma':'BRCA',
 'liver cancer':'LIHC','hepatocellular carcinoma':'LIHC','liver hepatocellular carcinoma':'LIHC',
 'lung adenocarcinoma':'LUAD','lung squamous cell carcinoma':'LUSC',
 'colon adenocarcinoma':'COAD','rectum adenocarcinoma':'READ',
 'prostate cancer':'PRAD','prostate adenocarcinoma':'PRAD','stomach adenocarcinoma':'STAD',
 'glioblastoma':'GBM','glioblastoma multiforme':'GBM','ovarian serous cystadenocarcinoma':'OV',
 'pancreatic adenocarcinoma':'PAAD','bladder urothelial carcinoma':'BLCA',
 'head and neck squamous cell carcinoma':'HNSC','thyroid carcinoma':'THCA',
}
def text(value):
    if value is None:return ''
    s=str(value).strip()
    return '' if s.lower() in {'nan','none','<na>','na','unknown','unspecified'} else s

def cancer_context(value):
    s=text(value);u=s.upper().removeprefix('CANCER:').removeprefix('TCGA-')
    if u in CANCERS:return u
    if u in {'PAN','PANCAN','PAN-CANCER','PAN_CANCER','ALL'}:return 'PAN_CANCER'
    if s.casefold() in ALIASES:return ALIASES[s.casefold()]
    return 'CONTEXT_UNRESOLVED' if s else 'CONTEXT_UNSPECIFIED'

def predicted_method(value):
    return bool(re.search(r'(?i)catrap(?:id)?|rpi(?:seq|pred)|rna.?protein.*predict|in.?silico|computational|prediction|predicted',text(value)))

def physical_method(value):
    s=text(value)
    if predicted_method(s):return False
    return bool(re.search(r'(?i)(?<![A-Za-z])(?:eCLIP|iCLIP|PAR-CLIP|HITS-CLIP|CLIP|ChIRP|CHART|RAP(?:-MS|-RNA)?|RIP|EMSA|co-IP|IP)(?![A-Za-z])|pull[ -]?down|immunoprecipitation|physical_binding|physical interaction',s))

def relation_kind(value):
    s=text(value).lower()
    if 'cerna' in s or 'sponge' in s:return 'ceRNA_mediated'
    if 'bind' in s:return 'binding'
    if 'interaction' in s:return 'interaction_unspecified'
    if 'regulat' in s or 'target' in s:return 'regulation'
    return 'association'

def experiment_kind(value):
    if predicted_method(value):return 'computational_prediction'
    if physical_method(value):return 'physical_binding'
    s=text(value).lower()
    if re.search(r'knock|sirna|shrna|crispr|overexpress|transfect|deplet',s):return 'functional_perturbation'
    if re.search(r'luciferase|reporter',s):return 'reporter_assay'
    if re.search(r'qpcr|rt-pcr|rna-seq|microarray|western',s):return 'expression_or_abundance'
    return 'other_experimental' if s else 'unspecified'
