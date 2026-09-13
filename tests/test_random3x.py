from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from random3x_runtime import selected_encounters,technical_valid,ScheduleSelector,resume_continuity


def group(e,correct=True):
    g=dict(e,group_id='new-'+str(e['encounter_index']))
    g['responses']=[dict(group_id=g['group_id'],prompt_id=e['prompt_id'],policy_version='newpolicy',config_sha256='c',reward_version='r',
        trajectory_id=g['group_id']+str(i),member_index=i,finish_reason='stop',acc=int(correct),parsed_answer='A' if correct else 'B',total_reward=1. if correct else 0.) for i in range(4)]
    return g


class RandomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schedule=json.loads((ROOT/'experiments/stage5/aux_random3x_dynamic_schedule_v1.json').read_text())['windows']

    def test_exact_frozen_schedule_and_identity_selection(self):
        selector=ScheduleSelector(self.schedule);count=0
        for i,w in enumerate(self.schedule):
            expected=sorted(w['encounters'],key=lambda e:hashlib.sha256(f'20260914:{i}:{e["encounter_index"]}:{e["prompt_id"]}'.encode()).hexdigest())[:8]
            selected=[];decisions=[]
            for start in range(0,len(w['encounters']),8):
                groups=[group(e,False) for e in w['encounters'][start:start+8]]
                ss,dd=selector(groups,'newpolicy','random3x');selected+=ss;decisions+=dd
                if start+8<len(w['encounters']):self.assertEqual(ss,[])
            self.assertEqual([g['prompt_id'] for g in selected],[e['prompt_id'] for e in expected])
            self.assertEqual(len(selected),8);self.assertEqual(len(decisions),w['generated_groups']);count+=len(decisions)
        self.assertEqual(count,1360)

    def test_selection_independent_of_all_outcomes(self):
        w=self.schedule[0];outputs=[]
        for correct in [False,True]:
            selector=ScheduleSelector(self.schedule);selected=[]
            for start in range(0,len(w['encounters']),8):
                gg=[group(e,correct) for e in w['encounters'][start:start+8]]
                ss,_=selector(gg,'newpolicy','random3x');selected.extend(ss)
            outputs.append([g['prompt_id'] for g in selected])
        self.assertEqual(outputs[0],outputs[1])

    def test_seed_determinism(self):
        e=self.schedule[0]['encounters']
        self.assertEqual(selected_encounters(e,0),selected_encounters(list(reversed(e)),0))
        self.assertNotEqual(selected_encounters(e,0),selected_encounters(e,0,20260915))

    def test_technical_failure_zero_retry(self):
        e=self.schedule[0]['encounters'];gg=[group(x) for x in e[:8]]
        gg[0]['responses'][0]['transport_error']='timeout'
        self.assertFalse(technical_valid(gg[0],'newpolicy'))
        with self.assertRaises(AssertionError):ScheduleSelector(self.schedule)(gg,'newpolicy','random3x')
        g=group(e[0]);g['responses'].pop();self.assertFalse(technical_valid(g,'newpolicy'))
        self.assertTrue(technical_valid(group(e[0],False),'newpolicy'))

    def test_resume_rejects_optimizer_replay(self):
        state=dict(training_groups=32,optimizer_steps=8);pause=dict(state=state,next_encounters=[1,2])
        restarted=deepcopy(pause);commit=dict(state_before=state,state_after=dict(training_groups=40,optimizer_steps=10))
        resume_continuity(pause,restarted,commit)
        bad=deepcopy(commit);bad['state_after']['optimizer_steps']=12
        with self.assertRaises(AssertionError):resume_continuity(pause,restarted,bad)

    def test_disjoint_frozen_eval_when_available(self):
        f=ROOT/'experiments/stage5/ablation_eval_512_v1.json'
        if not f.exists():self.skipTest('Run again after data freeze')
        m=json.loads(f.read_text());rows=json.loads(Path(m['dataset']['path']).read_text())
        self.assertEqual(len(rows),512);self.assertEqual(len({r['cluster_id'] for r in rows}),512)
        self.assertFalse(any(m['overlap_counts'].values()))
        p=json.loads((ROOT/'experiments/stage5/aux_random3x_protocol_v2.json').read_text())
        self.assertEqual(p['project_state_snapshot'],json.loads((ROOT/'project_state.json').read_text()))

    def test_real_retained_native_advantage_and_group_accounting(self):
        from verify_stage4 import update
        from analyze_signal_density import inspect_run
        pair=json.loads((ROOT/'experiments/stage4/formal_pair.json').read_text())
        path=Path(pair['runs']['vanilla']['path'])
        x=inspect_run(path,1)
        self.assertEqual(x['checks']['groups'],8)
        self.assertEqual(x['checks']['trajectories'],32)
        by=x['summary']['by_type']
        self.assertEqual(sum(by[k]['groups'] for k in ['mixed','all_correct','all_wrong']),8)


if __name__=='__main__':unittest.main()
