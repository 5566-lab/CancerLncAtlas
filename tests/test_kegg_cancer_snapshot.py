"""Offline checks for official KEGG snapshot completeness and identity safety."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'fetch_kegg_cancer_backbone.py'
spec = importlib.util.spec_from_file_location('fetch_kegg_cancer_backbone', SCRIPT)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def hierarchy():
    return {'name':'root', 'children':[
        {'name':'Cancer: overview','children':[{'name':'05200  Pathways in cancer'}]},
        {'name':'Cancer: specific types','children':[{'name':'05224  Breast cancer'}]},
        {'name':'Other cancer-related collection','children':[{'name':'00010  Glycolysis'}]},
    ]}


class KeggSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.paths=fetch.select_cancer_paths(hierarchy())
        self.names='hsa05200\tPathways in cancer - Homo sapiens (human)\nhsa05224\tBreast cancer - Homo sapiens (human)\n'
        self.links='path:hsa05200\thsa:1\npath:hsa05224\thsa:2\npath:hsa00010\thsa:3\n'
        self.entrez='hsa:1\tncbi-geneid:1\nhsa:2\tncbi-geneid:2\n'
        self.genes='hsa:1\tCDS\t1:1..10\tONE, ALIAS; protein one\nhsa:2\tncRNA\t2:1..10\tTWO; RNA two\n'

    def test_official_categories_exclude_name_only_cancer_matches(self):
        self.assertEqual([p['kegg_id'] for p in self.paths], ['hsa05200','hsa05224'])

    def test_absent_category_fails(self):
        tree=hierarchy(); tree['children'].pop(1)
        with self.assertRaisesRegex(ValueError,'Expected one official category'):
            fetch.select_cancer_paths(tree)

    def test_duplicate_category_fails(self):
        tree=hierarchy(); tree['children'].append(tree['children'][0])
        with self.assertRaisesRegex(ValueError,'Expected one official category'):
            fetch.select_cancer_paths(tree)

    def test_human_list_and_membership_and_official_id_mapping(self):
        rows,_=fetch.build_membership(self.paths,self.names,self.links,self.entrez,self.genes)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['gene_aliases'],'ONE|ALIAS')
        self.assertEqual(rows[1]['gene_biotype'],'ncRNA')
        self.assertEqual(rows[1]['entrez_gene_id'],'2')

    def test_reference_pathway_without_human_definition_fails(self):
        with self.assertRaisesRegex(ValueError,'absent from official human pathway list'):
            fetch.build_membership(self.paths,self.names.splitlines()[0],self.links,self.entrez,self.genes)

    def test_empty_pathway_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError,'without gene definitions'):
            fetch.build_membership(self.paths,self.names,self.links.splitlines()[0],self.entrez,self.genes)

    def test_unmapped_gene_cannot_be_assumed_entrez(self):
        with self.assertRaisesRegex(ValueError,'Missing verified Entrez ID'):
            fetch.build_membership(self.paths,self.names,self.links,self.entrez.splitlines()[0],self.genes)

    def test_duplicate_links_not_double_counted(self):
        rows,_=fetch.build_membership(self.paths,self.names,self.links+self.links,self.entrez,self.genes)
        self.assertEqual(len(rows),2)

    def test_existing_frozen_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=root/'source.txt'; path.write_text('official')
            manifest={'sources':[{'path':'source.txt','sha256':hashlib.sha256(b'official').hexdigest()}], 'outputs':[]}
            fetch.verify_existing(root,manifest)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'failed checksum'):
                fetch.verify_existing(root,manifest)

    def test_preparation_on_non149_host_fails_before_creating_outputs(self):
        with patch.object(fetch.socket,'gethostname',return_value='windows'):
            with self.assertRaisesRegex(RuntimeError,'must run on 149'):
                fetch.freeze('/does-not-exist')


if __name__=='__main__':
    unittest.main()
