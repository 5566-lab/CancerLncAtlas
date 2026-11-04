"""Source-defined State members and protein bridges, not inferred causal edges."""
from __future__ import annotations
import argparse, gzip, json, re, socket, tarfile
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, read_json, sha, digest, fail
from .state_catalog import AUGMENTED_CATALOG_ID, expanded_targets

FORMAT = 'CLA_STATE_PROTEIN_MAP_V1'


def pinned(rec):
    p = Path(rec['path']).resolve()
    if not p.is_file() or sha(p) != rec.get('sha256'):
        fail('STATE_MAP_SOURCE_HASH', 'Source missing or changed', path=str(p))
    return p


def reference_genes(gtf):
    genes = {}
    by_symbol = defaultdict(set)
    with gzip.open(gtf, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] != 'gene':
                continue
            a = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
            gid = re.sub(r'\.\d+(?=_PAR_Y$|$)', '', a['gene_id'])
            genes[gid] = {'symbol': a['gene_name'], 'biotype': a.get('gene_type', '')}
            by_symbol[a['gene_name']].add(gid)
    return genes, by_symbol


class ProteinResolver:
    def __init__(self, genes, by_symbol, hgnc, graph):
        self.genes, self.by_symbol = dict(genes), by_symbol
        self.gencode_ids = set(genes)
        self.entrez = defaultdict(set)
        self.hgnc = hgnc.fillna('')
        self.by_hgnc_id = defaultdict(list)
        self.symbols = defaultdict(set)
        self.aliases = defaultdict(set)
        for row in self.hgnc.to_dict('records'):
            gid = row['ensembl_gene_id']
            if not gid or row.get('status') != 'Approved':
                continue
            if gid not in self.genes:
                coding = row.get('locus_group')=='protein-coding gene' or row.get('locus_type')=='gene with protein product'
                self.genes[gid] = {'symbol':row['symbol'], 'biotype':'protein_coding' if coding else row.get('locus_type','HGNC_NONCODING')}
            if row.get('entrez_id'):self.entrez[str(row['entrez_id'])].add(gid)
            self.by_hgnc_id[gid].append(row)
            self.symbols[row['symbol']].add(gid)
            for field in ('prev_symbol', 'alias_symbol'):
                for alias in row.get(field, '').split('|'):
                    if alias:
                        self.aliases[alias].add(gid)
        self.graph = defaultdict(set)
        for row in graph.to_dict('records'):
            self.graph[str(row['gene_id']).removeprefix('GENE:')].add(str(row['protein_id']).removeprefix('UNIPROT:'))

    def resolve(self, symbol, explicit_gene=''):
        explicit_gene = str(explicit_gene or '')
        if explicit_gene:
            candidates = {explicit_gene} if explicit_gene in self.genes else set()
            basis = 'SOURCE_ENSEMBL_GENCODE_V36'
        else:
            candidates = self.by_symbol.get(symbol, set()) or self.symbols.get(symbol, set())
            basis = 'EXACT_REFERENCE_SYMBOL'
            if not candidates:
                candidates = self.aliases.get(symbol, set())
                basis = 'UNAMBIGUOUS_HGNC_PREVIOUS_OR_ALIAS_SYMBOL'
            if not candidates and re.fullmatch(r'LOC[0-9]+',symbol):
                candidates=self.entrez.get(symbol[3:],set())
                basis='EXACT_ENTREZ_FROM_LOC_IDENTIFIER'
        empty = {'gene_id': '', 'reference_symbol': '', 'gene_biotype': '', 'uniprot_accessions': '',
                 'current_graph_protein_ids': '', 'mapping_basis': basis, 'reference_presence':''}
        if len(candidates) != 1:
            return {**empty, 'mapping_status': 'AMBIGUOUS_SYMBOL' if candidates else 'UNRESOLVED_REFERENCE_GENE',
                    'candidate_gene_ids': '|'.join(sorted(candidates))}
        gid = next(iter(candidates))
        gene = self.genes[gid]
        common = {**empty, 'gene_id': gid, 'reference_symbol': gene['symbol'], 'gene_biotype': gene['biotype'],
                  'candidate_gene_ids': '', 'reference_presence':'GENCODE_V36' if gid in self.gencode_ids else 'HGNC_ONLY_NOT_IN_GENCODE_V36'}
        if gene['biotype'] != 'protein_coding':
            return {**common, 'mapping_status': 'NON_PROTEIN_CODING_MEMBER'}
        accessions = set()
        for row in self.by_hgnc_id[gid]:
            accessions.update(v for v in row.get('uniprot_ids', '').split('|') if v)
        # Graph links are existing model identifiers, not substituted for authoritative HGNC links.
        graph = self.graph.get(gid, set())
        return {**common, 'uniprot_accessions': '|'.join(sorted(accessions)),
                'current_graph_protein_ids': '|'.join('UNIPROT:'+v for v in sorted(graph)),
                'mapping_status': 'MAPPED_PROTEIN' if accessions else 'PROTEIN_CODING_WITHOUT_HGNC_UNIPROT'}


def collect_members(paths, targets):
    members = []
    target_by_id = {t['id']: t for t in targets}

    def add(sid, symbol='', gene='', weight=None, direction='', relation='SIGNATURE_MEMBER',
            component='', source='', note='', probe=''):
        if sid not in target_by_id:
            fail('STATE_MAP_UNKNOWN_TARGET', sid)
        members.append({'state_id': sid, 'source_symbol': str(symbol or ''), 'source_gene_id': str(gene or ''),
            'source_probe_id': probe, 'source_weight': weight, 'score_direction': direction,
            'relation_type': relation, 'signature_component': component, 'source_resource': source,
            'mapping_note': note})

    defs = read_json(paths['signifinder'])
    choices = {
        'Hypoxia_Buffa': ('Hypoxia_Buffa', None), 'EMT_Miow_Epithelial': ('EMT_Miow', 'Epithelial-like'),
        'EMT_Miow_Mesenchymal': ('EMT_Miow', 'Mesenchymal-like'), 'CellCycle_Lundberg': ('CellCycle_Lundberg', None),
        'MitoticIndex_Yang': ('MitoticIndex_Yang', None), 'CIN_Carter_70': ('CIN_Carter', None),
        'DNArep_Kang': ('DNArep_Kang', None), 'Glycolysis_Zhang': ('Glycolysis_Zhang', None),
        'LipidMetabolism_Zheng': ('LipidMetabolism_Zheng', None), 'VEGF_Hu': ('VEGF_Hu', None),
        'IFN_Ayers': ('IFN_Ayers', None), 'Tinflam_Ayers': ('Tinflam_Ayers', None), 'APM_Wang': ('APM_Wang', None),
        'TLS_Cabrita': ('TLS_Cabrita', None), 'LRRC15CAF_Dominguez': ('LRRC15CAF_Dominguez', None),
        'ECM_Chakravarthy_up': ('ECM_Chakravarthy', 'ECMup'),
        'ECM_Chakravarthy_down': ('ECM_Chakravarthy', 'ECMdown')}
    for name, (obj, subset) in choices.items():
        for row in defs[obj]:
            if subset is not None and row.get('class') != subset:
                continue
            cls = row.get('class', '')
            weight = row.get('coeff')
            relation = 'NORMALIZATION_REFERENCE' if cls == 'Housekeeping' else 'SIGNATURE_MEMBER'
            direction = 'POSITIVE' if weight is not None and weight > 0 else 'NEGATIVE' if weight is not None and weight < 0 else ''
            if obj == 'DNArep_Kang':
                direction = 'HIGH_EXPRESSION_INDICATOR' if cls == 'high' else 'LOW_EXPRESSION_INDICATOR'
            add('signifinder::'+name, row['SYMBOL'], weight=weight, direction=direction,
                relation=relation, component=cls, source='signifinder',
                note='Score direction is not biochemical activation/inhibition; cohort-fitted weights are not invented.')
    for g in read_json(paths['senmayo'])['SAUL_SEN_MAYO']['geneSymbols']:
        add('SenMayo', g, source='senmayo')
    hallmark = {}
    for line in paths['hallmark'].read_text().splitlines():
        fields = line.split('\t')
        if len(fields)>2:
            hallmark[fields[0]] = fields[2:]
    for sid, name in [('OXPHOS', 'HALLMARK_OXIDATIVE_PHOSPHORYLATION'), ('DNARepair_Hallmark', 'HALLMARK_DNA_REPAIR')]:
        for g in hallmark[name]:
            add(sid, g, source='hallmark')
    gene_sets = pd.read_excel(paths['panimmune_sets'], sheet_name='Genes')
    tls = gene_sets.loc[gene_sets.SetName.eq('Chemokine12_score'), 'Gene'].tolist()
    expected = {'CCL2','CCL3','CCL4','CCL5','CCL8','CCL18','CCL19','CCL21','CXCL9','CXCL10','CXCL11','CXCL13'}
    if set(tls) != expected or len(tls) != 12:
        fail('STATE_MAP_TLS12_DEFINITION', 'Official 12-chemokine membership changed')
    for g in tls:
        add('TLS12_Chemokine_Published', g, source='panimmune_sets', note='Published PanImmune/Wolf score; no reconstructed PCA coefficients.')
    malta = paths['malta']
    for sid, sheet, header in [('RNAss','mRNAsi',0), ('EREG.EXPss','EREG-mRNAsi',1)]:
        for row in pd.read_excel(malta, sheet_name=sheet, header=header).dropna(subset=['ENSEMBL']).to_dict('records'):
            add(sid, row['HUGO'], gene=row['ENSEMBL'], weight=float(row['Weight']), relation='OCLR_SIGNATURE_COEFFICIENT', source='malta')
    for sid, sheet in [('DNAss','mDNAsi'), ('EREG-METHss','EREG-mDNAsi'), ('DMPss','DMPsi'), ('ENHss','ENHsi')]:
        f = pd.read_excel(malta, sheet_name=sheet, header=1)
        for row in f.dropna(subset=[f.columns[0]]).itertuples(index=False, name=None):
            add(sid, weight=float(row[1]), relation='METHYLATION_PROBE_COMPONENT', probe=str(row[0]),
                source='malta', note='Original probe weight; no verified direct protein assignment.')
    with tarfile.open(paths['extend']) as z:
        match = [m for m in z.getmembers() if m.name.endswith('/R/IterativeRS.r')]
        if len(match)!=1:
            fail('STATE_MAP_EXTEND_SOURCE', 'Expected pinned original source')
        text = z.extractfile(match[0]).read().decode()
    genes = re.search(r'genes\s*=\s*c\(([^)]+)\)', text)
    if genes is None:
        fail('STATE_MAP_EXTEND_SOURCE', 'Original signature expression not found')
    for g in re.findall(r"['\"]([^'\"]+)['\"]", genes.group(1)):
        add('EXTEND', g, relation='SOURCE_CODE_SIGNATURE_TOKEN', source='extend',
            note='Original v1 source contains aliases/noncoding markers; tokens are not asserted to be distinct proteins or fitted coefficients.')
    hacksig = pd.DataFrame(read_json(paths['hacksig'])['signatures_data']).fillna('')
    for sid, signatures in [('ImmuneScore',['estimate_immune']), ('StromalScore',['estimate_stromal']),
        ('ESTIMATEScore',['estimate_immune','estimate_stromal']), ('TumorPurity',['estimate_immune','estimate_stromal']),
        ('IPS',['ips_mhc','ips_cp','ips_ec','ips_sc']), ('CYT',['rooney2015_cyt'])]:
        f = hacksig.loc[hacksig.signature_id.isin(signatures)]
        for row in f.to_dict('records'):
            weight = row.get('gene_weight') or None
            add(sid, row['gene_symbol'], weight=weight, source='hacksig',
                component=row['signature_id'], relation='DERIVED_SCORE_COMPONENT' if sid in ['TumorPurity','ESTIMATEScore'] else 'SIGNATURE_MEMBER',
                note='IPS category/marker weights are not a single gene-linear coefficient; purity is an inverse nonlinear transformation of ESTIMATE.')
    xcell = read_json(paths['xcell'])
    for name, gs in xcell['signatures'].items():
        cell = name.split('%',1)[0]
        sid = 'xCell::'+cell
        if sid not in target_by_id:
            fail('STATE_MAP_XCELL_CELL_ID', cell)
        if isinstance(gs, str): gs=[gs]
        for g in gs:
            add(sid, g, source='xcell', relation='CELL_ENRICHMENT_SUBSIGNATURE', component=name,
                note='Original xCell membership; final score also uses calibration and spillover correction.')
    mcp = pd.read_csv(paths['mcp'], sep='\t')
    for row in mcp.to_dict('records'):
        sid = 'MCPcounter::'+row['Cell population'].replace(' ', '.')
        # Historical score IDs may preserve spaces; match only deterministic R name conversion.
        matches = [t['id'] for t in targets if t['id'].startswith('MCPcounter::') and
                   t['id'].split('::',1)[1].replace(' ','.') == row['Cell population'].replace(' ','.')]
        if len(matches)!=1: fail('STATE_MAP_MCP_CELL_ID', sid)
        add(matches[0], row['HUGO symbols'], gene=str(row['ENSEMBL ID']), source='mcp', relation='CELL_ABUNDANCE_MARKER')
    if 'lm22' in paths:
        lm22 = pd.read_csv(paths['lm22'], sep='\t')
        lm22_degs = pd.read_csv(paths['lm22_degs'], sep='\t').set_index('gene_symbol')
        for cell in lm22.columns[1:]:
            sid = 'cibersort_relative::'+re.sub(r'[^A-Za-z0-9_.]', '.', cell)
            if sid not in target_by_id:fail('STATE_MAP_LM22_CELL_ID', sid)
            for g, value in zip(lm22.iloc[:,0], lm22[cell]):
                add(sid, g, weight=float(value), source='lm22', relation='CELL_REFERENCE_EXPRESSION',
                    component='CELL_SPECIFIC_DEG' if int(lm22_degs.loc[g,cell])==1 else 'JOINT_REFERENCE_NONDEG', note='Reference expression is not a score coefficient or a cell-specific causal marker; deconvolution uses the joint reference matrix.')
    for role, label, gene_col, weight_col in [('progeny','PROGENY:','gene','weight'), ('dorothea','DOROTHEA:','target','mor')]:
        f = pd.read_csv(paths[role], sep='\t')
        f['state'] = label+f['pathway' if role=='progeny' else 'tf']
        f = f.loc[f.state.isin(target_by_id) & np.isfinite(pd.to_numeric(f[weight_col], errors='coerce'))]
        # Reproduce the existing R/07 weighted-resource aggregation exactly; do not introduce top-N filtering.
        f = f.groupby(['state', gene_col], sort=True)[weight_col].mean().reset_index()
        for sid, g, weight in f.itertuples(index=False, name=None):
            add(sid, g, weight=float(weight), direction='POSITIVE' if weight>0 else 'NEGATIVE' if weight<0 else 'ZERO',
                source=role, relation='PATHWAY_RESPONSE_FOOTPRINT' if role=='progeny' else 'TF_REGULON_TARGET',
                note='Existing full-resource weighted-score definition; not a pathway physical-membership claim.')
    return pd.DataFrame(members)


def build(spec, output):
    if socket.gethostname().split('.')[0] != '149':
        fail('STATE_EXECUTION_HOST', 'Run mapping preparation on 149')
    paths = {k:pinned(v) for k,v in spec['sources'].items()}
    targets = expanded_targets(AUGMENTED_CATALOG_ID)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        fail('STATE_OUTPUT_EXISTS', 'Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json', {'target_host':'149','operation':'STATE_PROTEIN_SIGNATURE_MAPPING'})
    members = collect_members(paths, targets)
    genes, by_symbol = reference_genes(paths['gtf'])
    resolver = ProteinResolver(genes, by_symbol, pd.read_csv(paths['hgnc'], sep='\t', dtype=str),
                              pd.read_parquet(paths['protein_graph']))
    keys = members[['source_symbol','source_gene_id']].drop_duplicates()
    bridge = pd.DataFrame([{**row, **resolver.resolve(row['source_symbol'],row['source_gene_id'])}
                          for row in keys.to_dict('records')])
    mapped = members.merge(bridge, on=['source_symbol','source_gene_id'], how='left', validate='many_to_one')
    probes = mapped.source_probe_id.ne('')
    mapped.loc[probes, 'mapping_status'] = 'PROBE_ONLY_NO_VERIFIED_PROTEIN_BRIDGE'
    rppa = pd.read_csv(paths['rppa_catalog'], sep='\t', keep_default_na=False)
    coverage = pd.read_csv(paths['rppa_coverage'], sep='\t')
    # RPPA bridge labels multi-gene antibodies explicitly; it never implies gene-specific measured abundance.
    rppa_map = defaultdict(list)
    for row in rppa.to_dict('records'):
        for gid in row['gene_ids'].split('|'):
            if gid:
                rppa_map[gid].append(row['feature_id'])
    mapped['rppa500_feature_ids'] = mapped.gene_id.map(lambda g:'|'.join(sorted(rppa_map.get(g,[]))))
    observed = set(coverage.loc[coverage.observed_patients.gt(0), 'feature_id'])
    mapped['rppa500_features_with_any_measurement'] = mapped.gene_id.map(lambda g:'|'.join(sorted(set(rppa_map.get(g,[])) & observed)))
    mapped['rppa500_bridge_role'] = 'ANNOTATED_TARGET_ONLY;MULTIGENE_ANTIBODY_NOT_GENE_SPECIFIC;SEE_RPPA_FEATURE_CATALOG'
    covered = set(mapped.state_id)
    for t in targets:
        if t['id'] not in covered:
            missing = 'REFERENCE_RESOURCE_NOT_RECOVERED' if t['id'].startswith('cibersort_relative::') else 'NO_FIXED_GENE_OR_PROTEIN_SIGNATURE'
            mapped = pd.concat([mapped,pd.DataFrame([{'state_id':t['id'],'mapping_status':missing,
                'mapping_note':'Score is not defined by a fixed protein list; no mechanistic gene list substituted.'}])],ignore_index=True)
    meta = pd.DataFrame([{'state_id':t['id'],'state_family':t['family'],'score_method':t['method'],
        'measurement_layer':t['measurement'],'state_role':t['role'],'score_source':t['source']} for t in targets])
    mapped = mapped.merge(meta, on='state_id', validate='many_to_one')
    mapped = mapped.sort_values(['state_id','source_symbol','signature_component','source_probe_id'],na_position='last').reset_index(drop=True)
    path = out/'STATE_PROTEIN_MAP.tsv.gz'
    mapped.to_csv(path, sep='\t', index=False, compression={'method':'gzip','mtime':0},na_rep='')
    summary=[]
    for t in targets:
        f=mapped.loc[mapped.state_id.eq(t['id'])]
        summary.append({'state_id':t['id'],'family':t['family'],'method':t['method'],'role':t['role'],
            'rows':len(f),'source_symbols':f.source_symbol.replace('',np.nan).nunique(),
            'mapped_genes':f.gene_id.replace('',np.nan).nunique(),
            'mapped_protein_genes':f.loc[f.mapping_status.eq('MAPPED_PROTEIN'),'gene_id'].nunique(),
            'mapping_statuses':'|'.join(sorted(f.mapping_status.unique())),
            'source_resources':'|'.join(sorted(f.source_resource.dropna().unique()))})
    pd.DataFrame(summary).to_csv(out/'STATE_MAPPING_SUMMARY.tsv',sep='\t',index=False)
    unresolved=mapped.loc[~mapped.mapping_status.eq('MAPPED_PROTEIN') & mapped.source_symbol.fillna('').ne('')]
    unresolved.to_csv(out/'UNRESOLVED_OR_NONCODING_MEMBERS.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    # Reusing source checksums also verifies that no resource drift occurred during the mapping run.
    for key, rec in spec['sources'].items():pinned(rec)
    atomic_json(out/'SOURCE_SPEC.json',spec)
    atomic_json(out/'COLUMN_MEANINGS.json',{'source_weight':'Native signed coefficient, OCLR weight, IPS category weight or reference expression according to relation_type; blank if no fixed coefficient exists.',
        'uniprot_accessions':'HGNC curated UniProt accessions; reference_presence separates GENCODE v36 loci from HGNC-only loci. Not proof of measured protein.',
        'current_graph_protein_ids':'Existing model identifiers, recorded separately from HGNC links.',
        'rppa500_feature_ids':'Antibodies annotated to this gene; multi-gene antibodies and missing measurements require the RPPA catalog and coverage.',
        'relation_type':'Definition of a scoring dependency or prior; no new causal interaction is inferred.',
        'mapping_status':'Every State is accounted for, including probe-only, noncoding, ambiguous and intrinsically non-gene measurements.'})
    report={'format':FORMAT,'status':'COMPLETE','target_host':'149','catalog_id':AUGMENTED_CATALOG_ID,'producer_sha256':sha(Path(__file__)),
        'target_catalog_sha256':digest(targets),'states':len(targets),'rows':len(mapped),
        'states_with_protein_mapping':int(mapped.loc[mapped.mapping_status.eq('MAPPED_PROTEIN'),'state_id'].nunique()),
        'unique_mapped_genes':int(mapped.gene_id.replace('',np.nan).nunique()),
        'mapping_status_counts':mapped.mapping_status.value_counts().to_dict(),'sources':spec['sources'],
        'used_as_training_graph_edges':False,'scores_changed':False,
        'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    atomic_json(out/'COMPLETE.json',report)
    return report


def validate_binding(binding, targets):
    if binding is None:return None
    p=pinned(binding['complete']);r=read_json(p)
    if (r.get('format')!=FORMAT or r.get('status')!='COMPLETE' or r.get('target_host')!='149' or
            r.get('target_catalog_sha256')!=digest(targets) or r.get('states')!=len(targets)):
        fail('STATE_MAP_CATALOG_DRIFT','Mapping does not match active State definitions')
    for rec in r['sources'].values():pinned(rec)
    for name,rec in r['files'].items():
        fp=(p.parent/name).resolve()
        if not fp.is_relative_to(p.parent) or not fp.is_file() or sha(fp)!=rec['sha256'] or fp.stat().st_size!=rec['bytes']:
            fail('STATE_MAP_PRODUCT_DRIFT',name)
    return {'status':'STATE_PROTEIN_MAP_VALIDATED_NOT_TRAINING_EDGES','complete':{'path':str(p),'sha256':sha(p)},
            'states':r['states'],'rows':r['rows'],'states_with_protein_mapping':r['states_with_protein_mapping']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build(read_json(a.source_spec),a.output),default=str,indent=2))


if __name__=='__main__':main()
