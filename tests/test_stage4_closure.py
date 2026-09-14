"""Real retained-artifact checks; synthetic rows only test future ranking rules."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
from checkpoint_selection import GROUPS, choose, freeze_json, ranking_key, read, reference, require_stage4_gate, validate_selection_input
from medical_posttrain.reward.parser import parse

S4 = ROOT/'experiments/stage4'
S5 = ROOT/'experiments/stage5'


class ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = read(S4/'manual_review_packet_v1.json')
        cls.protocol = read(S5/'checkpoint_selection_protocol_v1.json')
        cls.handoff = read(ROOT/'experiments/handoffs/stage4_final_docs_to_chatgpt_v1.json')

    def test_packet_exact_raw_responses_and_parser(self):
        count = 0
        cache = {}
        for case in self.packet['cases']:
            self.assertTrue(case['question'])
            self.assertTrue(case['options'])
            self.assertIn(case['source_split'].split('/')[0], ['cmexam_train','cmexam_val'])
            self.assertFalse(case['human_reviewed'])
            self.assertFalse(case['read_in_full'])
            self.assertIsNone(case['manual_observation'])
            for model, bundle in case['models'].items():
                if not bundle['available']:
                    self.assertTrue(bundle['reason']); continue
                for r in bundle['responses']:
                    path = r['raw_source']['path']
                    if path not in cache:
                        self.assertEqual(reference(path),r['raw_source'])
                        cache[path] = read(path)
                    source = cache[path]
                    if 'predictions' in source:
                        original = next(x for x in source['predictions'] if x['prompt_id']==case['prompt_id'])
                    else:
                        group = next(g for g in source['groups'] if g['prompt_id']==case['prompt_id'])
                        original = next(x for x in group['responses'] if x['trajectory_id']==r['trajectory_id'])
                        self.assertEqual(original['question'],case['question'])
                        self.assertEqual(original['options'],case['options'])
                        self.assertEqual(original['ground_truth'],case['ground_truth'])
                    self.assertEqual(r['visible_response'],original.get('raw_output',original.get('output')))
                    self.assertEqual(r['output_length_tokens'],len(original.get('token_ids',original.get('response_ids'))))
                    result = parse(r['visible_response'],''.join(case['options']),r['finish_reason'])
                    self.assertEqual(json.loads(json.dumps(result.to_dict())),r['parsed'])
                    self.assertEqual(r['correctness'],int(result.answer_set==case['ground_truth']))
                    count += 1
        self.assertEqual(count,84)  # 5x4 auxiliary, 4x3x4 Frontier, 4x4 training.

    def test_bidirectional_categories_and_complete_frontier(self):
        categories = {c['category']:c for c in self.packet['cases']}
        self.assertEqual(len(categories),13)
        for category in ['dynamic_correct_random3x_wrong','random3x_correct_dynamic_wrong',
                         'vanilla_correct_dynamic_wrong','rejected_all_correct','rejected_all_wrong','overflow_eligible','selected']:
            self.assertIn(category,categories)
        for c in self.packet['cases']:
            if 'Frontier' in c['source_split']:
                for m in ['sft','vanilla','dynamic']:
                    self.assertEqual(len(c['models'][m]['responses']),4)
                    self.assertEqual(sum(r['correctness'] for r in c['models'][m]['responses']),c['models'][m]['correct_count'])
        for cat, pattern in [('dynamic_correct_random3x_wrong',(1,0)),('random3x_correct_dynamic_wrong',(0,1))]:
            c = categories[cat]
            self.assertEqual(tuple(c['models'][m]['responses'][0]['correctness'] for m in ['dynamic','random3x']),pattern)
        self.assertGreaterEqual(categories['rejected_all_correct']['models']['dynamic']['output_tokens'],1024)

    def test_selected_native_advantages_and_source_refs(self):
        import numpy as np
        c = next(c for c in self.packet['cases'] if c['category']=='selected')
        for r in c['sources']:
            self.assertEqual(reference(r['path']),r)
        sel = read(next(r['path'] for r in c['sources'] if r['path'].endswith('selection.json')))['groups']
        bundle = c['models']['dynamic']
        offset = next(i for i,g in enumerate(sel) if g['group_id']==bundle['group_id'])*4
        with np.load(next(r['path'] for r in c['sources'] if r['path'].endswith('old.npz'))) as arrays:
            self.assertEqual([r['advantage'] for r in bundle['responses']],arrays['advantages'][offset:offset+4,0].astype(float).tolist())

    def test_no_selection_or_test_cases(self):
        ids = set(self.protocol['dataset']['prompt_ids'])
        for c in self.packet['cases']:
            self.assertNotIn(c['prompt_id'],ids)
            self.assertTrue(c['prompt_id'].startswith(('cmexam_train:','cmexam_val:')))
            self.assertNotIn('cmexam_test:',json.dumps(c))
        self.assertFalse(self.protocol['generation_performed'])
        self.assertTrue(self.handoff['selection1024_untouched'])
        self.assertTrue(self.handoff['final_tests_untouched'])

    def test_exact_candidates_and_real_adapter_hashes(self):
        index = read(S4/'formal_checkpoint_index.json')
        for v in ['vanilla','dynamic']:
            candidates = self.protocol['candidates'][v]
            self.assertEqual([c['training_groups'] for c in candidates],GROUPS)
            by_count = {r['training_groups']:r for r in index[v]['validation_checkpoints']}
            for c in candidates:
                self.assertEqual(c['adapter'],by_count[c['training_groups']]['adapter'])
                self.assertEqual(reference(c['adapter']['path']),c['adapter'])
                self.assertEqual(c['optimizer_steps'],c['policy_windows']*2)
            self.assertEqual(self.protocol['scientific_endpoints'][v],candidates[-1])

    def test_selection_partition_unchanged(self):
        r = self.protocol['dataset']['partition']
        self.assertEqual(reference(r['path']),r)
        part = read(r['path'])
        ids = self.protocol['dataset']['prompt_ids']
        self.assertEqual(ids,part['selection'])
        self.assertEqual(len(ids),1024)
        self.assertEqual(len(set(self.protocol['dataset']['cluster_ids'])),1024)
        self.assertFalse(set(ids)&set(part['monitor']))
        validate_selection_input(self.protocol,'cmexam_val_selection1024',ids)
        for split in ['cmexam_test','cmb_exam','Frontier1000','cmexam_val']:
            with self.assertRaises(ValueError):
                validate_selection_input(self.protocol,split,ids)
        with self.assertRaises(ValueError):
            validate_selection_input(self.protocol,'cmexam_val_selection1024',list(reversed(ids)))

    def test_all_ranking_tiebreaks(self):
        base = dict(checkpoint_id='b',training_groups=1024,n=1024,correct=600,unparseable=2,truncated=2)
        pairs = [(dict(correct=601,unparseable=9),{}),
                 (dict(unparseable=1,truncated=9),{}),
                 (dict(truncated=1,training_groups=5000),{}),
                 (dict(training_groups=512,checkpoint_id='z'),{}),
                 (dict(checkpoint_id='a'),{})]
        for winner,loser in pairs:
            a,b = base|winner,base|loser
            self.assertEqual(min([a,b],key=ranking_key),a)
            self.assertEqual(min([b,a],key=ranking_key),a)
        for field in ['test_score','response_length','Frontier_score','Random3x_result','CMB_score']:
            with self.assertRaises(ValueError): ranking_key(base|{field:1})
        with self.assertRaises(ValueError): ranking_key(base|{'n':512})

    def test_gate_blocks_current_real_stage(self):
        self.assertEqual(read(ROOT/'project_state.json')['stages']['4']['status'],'FULL_PASS')
        with self.assertRaises(PermissionError): require_stage4_gate()
        with self.assertRaises(PermissionError):
            choose(self.protocol,'vanilla',[],'cmexam_val_selection1024',self.protocol['dataset']['prompt_ids'])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            freeze_json(p/'project_state.json',{'stages':{'4':{'status':'DONE','verification_receipt':None}}})
            with self.assertRaises(PermissionError): require_stage4_gate(p)

    def test_freeze_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'protocol.json'; freeze_json(p,{'a':1}); original=p.read_bytes()
            with self.assertRaises(FileExistsError): freeze_json(p,{'a':2})
            self.assertEqual(p.read_bytes(),original)

    def test_exact_schema_and_no_fabricated_deliverables(self):
        schema = read(S4/'stage4_deliverables_schema_v1.json')
        self.assertEqual(reference(schema['verifier']['path']),schema['verifier'])
        text = Path(schema['verifier']['path']).read_text()
        self.assertIn(schema['exact_code'],text)
        self.assertEqual(schema['minimum_manual_reviews'],2)
        self.assertTrue(schema['consistent_with_existing_readiness'])
        self.assertFalse((S4/'deliverables.json').exists())
        self.assertFalse((ROOT/'docs/stage_reports/04_gspo.md').exists())

    def test_all_previous_artifacts_unchanged(self):
        start = read(S4/'closure_start_audit_v1.json')
        for r in start['protected_existing_files']:
            self.assertEqual(reference(r['path']),r)
        self.assertEqual(reference(start['project_state']['path']),start['project_state'])
        for stage in ['5','6']:
            self.assertEqual(read(ROOT/'project_state.json')['stages'][stage]['status'],'NOT_STARTED')

    def test_recovery_facts_and_handoff(self):
        r = read(S4/'recovery_interview_case_v1.json')
        self.assertFalse(r['human_reviewed'])
        self.assertEqual(r['optimizer_steps_completed'],[805,806])
        self.assertEqual(r['adoption']['optimizer_steps_reexecuted'],0)
        self.assertEqual(r['next_fresh_window']['optimizer_steps'],808)
        for source in r['sources']:
            self.assertEqual(reference(source['path']),source)
        self.assertEqual(self.handoff['random3x']['mechanism_status'],'INCONCLUSIVE')
        self.assertEqual(self.handoff['formal_pair']['observations']['final_delta_pp'],0.9765625)
        self.assertEqual(self.handoff['README_status'],'README_STATUS_STALE')
        self.assertEqual(self.handoff['status'],'WAITING_FOR_HUMAN_NARRATIVE_AND_MANUAL_REVIEW')
        for key in ['manual_review_packet','recovery_case','deliverables_schema','selection_protocol']:
            source = self.handoff[key]
            self.assertEqual(reference(source['path']),source)


if __name__=='__main__':
    unittest.main()
