"""Current documentation wiring, without executing a model or full stage verifier."""
import datetime
import json
from pathlib import Path
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_selection import read,reference,require_stage4_gate
from check_stage4_documentation import HANDOFF,PROMPT,REPORT,STORY,S4,verify


class DocumentationTests(unittest.TestCase):
    def test_consistency_and_all_protected_evidence(self):
        result=verify()
        self.assertEqual(result['result'],'PASS')
        self.assertEqual(result['numeric_claims_checked'],51)
        self.assertFalse(result['full_verifier_run'])

    def test_configuration_claims(self):
        h=read(HANDOFF);text=REPORT.read_text()
        self.assertIn(h['runtime_model']['base_revision'],text)
        self.assertIn(h['sft_initialization']['adapter_sha256'],text)
        for target in h['runtime_model']['LoRA']['lora_targets']:
            self.assertIn(target,text)
        for literal in ['r=32','alpha=64','46068 MiB','LR=1e-6','3e-4 / 4e-4','ppo_epochs=1']:
            self.assertIn(literal,text)
        for variant in ['vanilla','dynamic']:
            self.assertIn(h['formal_pair']['runs'][variant]['run_id'],text)

    def test_unsupported_claims_remain_separate(self):
        text=REPORT.read_text()
        supported=text.split('### Supported conclusions\n',1)[1].split('### Unsupported conclusions\n',1)[0]
        unsupported=text.split('### Unsupported conclusions\n',1)[1].split('### Acceptance status\n',1)[0]
        for phrase in ['statistically significantly beats','globally compute efficient','experimentally proven better',
                       'causal selection mechanism is proven','mixed-resolution H1 is supported','clinical validity','final held-out test superiority']:
            self.assertNotIn(phrase,supported)
            self.assertIn(phrase,unsupported)
        self.assertIn('INCONCLUSIVE',supported)
        for phrase in ['Negative result retention','single training seed','NOT SUPPORTED','CI']:
            self.assertIn(phrase.lower(),text.lower())

    def test_interview_structure_and_grpo_gspo_distinction(self):
        text=STORY.read_text()
        for heading in ['## 30-second version','## 90-second version']+[f'### Q{i}.' for i in range(1,9)]:
            self.assertIn(heading,text)
        for phrase in ['GRPO-style','group-relative','GSPO','sequence-level','formal ablation','INCONCLUSIVE']:
            self.assertIn(phrase,text)
        self.assertIn('项目没有 GRPO-loss vs GSPO-loss formal ablation',text)

    def test_drafts_have_no_confirmation(self):
        manual=read(S4/'formal_manual_cases_draft_v1.json')
        self.assertEqual(manual['manual_reviews'],[])
        self.assertEqual(manual['status'],'WAITING_FOR_USER_CONFIRMATION')
        self.assertEqual(manual['human_review_count'],0)
        owner_text=PROMPT.read_text()
        for review in manual['proposed_reviews']:
            self.assertIn(review['suggested_observation'],owner_text)
            for flag in ['human_reviewed','read_in_full','user_confirmed']:
                self.assertIs(review[flag],False)
            self.assertEqual(len(review['sources']),4)
            for source in review['sources']:self.assertEqual(reference(source['path']),source)
        self.assertEqual([r['case_id'] for r in manual['proposed_reviews']],['S4-MR-02','S4-MR-03'])
        draft=read(S4/'deliverables_draft_v1.json')
        self.assertEqual(draft['READY_FOR_STAGE5'],'NO')
        self.assertFalse((S4/'deliverables.json').exists())
        self.assertFalse((S4/'formal_manual_cases_v1.json').exists())

    def test_gate_and_no_selection_or_final_test_results(self):
        protocol=read(ROOT/'experiments/stage5/checkpoint_selection_protocol_v1.json')
        self.assertFalse(protocol['generation_performed'])
        with self.assertRaises(PermissionError):require_stage4_gate()
        state=read(ROOT/'project_state.json')
        self.assertEqual(state['stages']['4']['status'],'FULL_PASS')
        for stage in ['5','6']:self.assertEqual(state['stages'][stage]['status'],'NOT_STARTED')
        self.assertTrue(all(value==0 for value in state['stages']['5']['progress'].values()))
        pair=read(S4/'formal_pair.json')
        for run in pair['runs'].values():
            for f in (Path(run['path'])/'validation').glob('*/summary.json'):
                row=read(f)
                self.assertFalse(row['test_used'])
                self.assertFalse(row['selection_used'])
        # No new bulk model artifacts since this task's input audit. This scans
        # only filesystem metadata, including abandoned/unindexed attempts.
        began=datetime.datetime.fromisoformat(read(S4/'documentation_input_audit_v1.json')['timestamp']).timestamp()
        bulk=Path('/data/WSH/medical-post-train-artifacts/runs')
        names={'raw.json','predictions.json','reservation.json','actual_policies.json'}
        new=[]
        for f in bulk.rglob('*.json'):
            if (f.name in names or f.name.endswith('_reservation.json')) and f.stat().st_mtime>began:
                new.append(str(f))
        self.assertEqual(new,[])
        # Auxiliary results remain disjoint and are not final-test outputs.
        aux=read(ROOT/'experiments/stage5/aux_random3x_eval_analysis_v1.json')
        self.assertFalse(aux['selection1024_used'])
        self.assertFalse(aux['final_test_used'])

    def test_all_new_document_links_resolve(self):
        for path in [REPORT,STORY]:
            for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                self.assertFalse(target.startswith(('http:','https:')))
                self.assertTrue((path.parent/target).resolve().is_file(),target)

    def test_readme_change_is_limited_to_status(self):
        import subprocess
        old=subprocess.check_output(['git','show','8a5d137:README.md'],cwd=ROOT,text=True)
        new=(ROOT/'README.md').read_text()
        before='The complete [Random-3x v2 auxiliary protocol]'
        after='A GPU-release assertion stopped training'
        self.assertEqual(old.split(before)[0],new.split('The [Random-3x auxiliary run]')[0])
        old_tail=old[old.index(after):].replace('The persistent queue remains active. See the',
             'Both formal runs subsequently completed; the training queue is no longer active. See the')
        self.assertEqual(old_tail,new[new.index(after):])
        for phrase in ['63.87%','62.50%','64.65%','INCONCLUSIVE','READY_FOR_STAGE5=NO','zero']:
            self.assertIn(phrase,new)


if __name__=='__main__':unittest.main()
