"""Auxiliary objective contract tests (no model generation)."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from loss_objective_metrics import length_bin,select

def test_length_boundaries():
    assert [length_bin(n) for n in [191,192,255,256,383,384,1024]]==['<192','[192,256)','[192,256)','[256,384)','[256,384)','>=384','>=384']

def test_selection_fixed_order():
    rows=[dict(learning_rate=lr,clip=c,health=dict(healthy=True,active_clipping=True)) for lr in [1e-5,1e-6,3e-6] for c in [.1,.2]]
    assert select(rows,{})['learning_rate']==1e-6
    assert select(rows,{})['clip']==.2
    rows[2]['health']['healthy']=rows[3]['health']['healthy']=False
    assert select(rows,{})['learning_rate']==3e-6

def test_native_token_objective_is_not_sequence_alias():
    import torch
    from verl.trainer.ppo.core_algos import get_policy_loss_fn,compute_policy_loss_vanilla,compute_policy_loss_gspo,compute_grpo_outcome_advantage
    from verl.workers.config import ActorConfig
    assert get_policy_loss_fn('vanilla') is compute_policy_loss_vanilla
    assert compute_policy_loss_vanilla is not compute_policy_loss_gspo
    cfg=ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,clip_ratio_low=.2,clip_ratio_high=.2)
    old=torch.zeros(1,2);cur=torch.tensor([[.5,-.5]],requires_grad=True);a=torch.ones(1,2);mask=torch.ones(1,2)
    token,_=compute_policy_loss_vanilla(old,cur,a,mask,config=cfg,loss_agg_mode='seq-mean-token-mean')
    seq,_=compute_policy_loss_gspo(old,cur,a,mask,config=cfg)
    assert abs(float(token.detach()-seq.detach()))>.05
    tg=torch.autograd.grad(token,cur,retain_graph=True)[0];sg=torch.autograd.grad(seq,cur)[0]
    assert not torch.allclose(tg,sg)
    import ast
    for path in ['scripts/loss_objective_actor.py','src/medical_posttrain/rl/actor.py']:
        src=(ROOT/path).read_text()
        assert 'compute_grpo_outcome_advantage(rewards,mask,np.repeat(np.arange(8),4),' in src
        assert src.index("immutable(directory/'old_frozen.json'")<src.index('for start in range(0,32')

def test_frozen_grid_and_diagnostic_integration():
    import json
    read=lambda p:json.loads(Path(p).read_text())
    pre=read(ROOT/'experiments/stage4/grpo_diagnostic_preregistration_v1.json')
    d=read(ROOT/'experiments/stage4/grpo_optimization_diagnostic_v1.json')
    assert pre['timestamp']<d['timestamp'] and len(d['conditions'])==6
    assert [(c['learning_rate'],c['clip']) for c in d['conditions']]==[(x['learning_rate'],x['clip']) for x in pre['grid']]
    for c in d['conditions']:
        if c.get('native_replay')!='PASS':continue
        assert read(Path(c['directory'])/'checkpoint/COMMITTED.json')['optimizer_step']==2
        assert c['metrics']['minibatches'][0]['ratio']['min']==c['metrics']['minibatches'][0]['ratio']['max']==1

def test_isolation_and_schedule():
    import json
    from medical_posttrain.rl.common import record,encounter
    from medical_posttrain.sampling.dynamic import Stream
    from medical_posttrain.evidence.stage2 import jsonlines
    read=lambda p:json.loads(Path(p).read_text())
    manifest=read(ROOT/'experiments/stage4/loss_objective_eval_manifest_v1.json')
    assert manifest['count']==512 and len(set(manifest['cluster_ids']))==512
    assert not any(manifest['overlap_counts'].values())
    assert not manifest['test_content_read'] and not manifest['selection_content_read']
    assert record(manifest['dataset']['path'])==manifest['dataset']
    schedule=read(ROOT/'experiments/stage4/loss_objective_vanilla_schedule_v1.json')
    cfg=read(ROOT/'configs/stages/s4_formal_shared.json')
    stream=Stream([r['prompt_id'] for r in jsonlines(cfg['pool']['path'])],cfg['seed'],cfg['stream_domain'])
    assert len(schedule['windows'])==64
    for i,w in enumerate(schedule['windows']):
        assert len(w['encounters'])==8
        for j,e in enumerate(w['encounters']):
            expected=encounter(stream,i*8+j,'unused','unused')
            assert all(e[k]==expected[k] for k in e)

def test_correctness_only_eligibility():
    import json,copy
    from medical_posttrain.rl.controller import select_groups
    p=json.loads((ROOT/'experiments/stage4/grpo_diagnostic_preregistration_v1.json').read_text())
    gs=json.loads(Path(p['original_batch']['path']).read_text())['groups']
    gs=copy.deepcopy(gs);policy=gs[0]['responses'][0]['policy_version']
    for count in range(5):
        g=copy.deepcopy(gs[0])
        for i,r in enumerate(g['responses']):r['acc']=int(i<count)
        a,ds=select_groups([g],policy,'dynamic')
        expected=0<count<4
        assert bool(a)==expected
        for r in g['responses']:
            for key in ['semantic','format','total_reward']:r[key]=1000
        b,es=select_groups([g],policy,'dynamic')
        assert ds==es and bool(b)==expected

def test_paired_statistics_and_interaction():
    from evaluate_loss_objective import paired
    a=[dict(prompt_id=str(i),acc=v) for i,v in enumerate([0,0,1,1])]
    b=[dict(prompt_id=str(i),acc=v) for i,v in enumerate([1,1,0,1])]
    x=paired(a,b);assert x==paired(a,b)
    assert x['wrong_to_correct']==2 and x['correct_to_wrong']==1
    assert x['bootstrap']['delta']==.25 and x['exact_mcnemar']==1
    # Per-prompt interaction averaging must equal the contrast of accuracies.
    vg=np.array([0,1,1,0]);dg=np.array([1,1,1,0]);vr=np.array([0,0,1,0]);dr=np.array([1,0,1,1])
    assert np.mean(dg-vg-dr+vr)==(dg.mean()-vg.mean())-(dr.mean()-vr.mean())

def test_budget_preregistration_and_stage_guards():
    import json,subprocess,hashlib
    from medical_posttrain.rl.common import record
    read=lambda p:json.loads(Path(p).read_text())
    path=ROOT/'experiments/stage4/grpo_objective_protocol_v1.json';p=read(path)
    assert p['budget']==dict(training_groups=512,accepted_mixed_dynamic=512,windows=64,optimizer_steps=128,G=4,training_trajectories=2048)
    for arm,r in p['runs'].items():
        c=read(r['config']['path']);assert c['target_training_groups']==512 and c['groups_per_window']==8 and c['mini_prompts']==4 and c['ppo_epochs']==1
        assert c['learning_rate']==p['selected']['learning_rate'] and c['clip_ratio_low']==c['clip_ratio_high']==p['selected']['clip']
        assert c['initialization']==p['config']['initialization']
        assert c['sampling_mode']==arm and c['pause_after_windows']==4
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(path.relative_to(ROOT))],cwd=ROOT,text=True).strip()
    blob=subprocess.check_output(['git','show',commit+':'+str(path.relative_to(ROOT))],cwd=ROOT)
    assert hashlib.sha256(blob).hexdigest()==record(path)['sha256']
    launch=read(ROOT/'experiments/stage4/loss_objective_launch_v1.json')
    assert launch['timestamp']>p['timestamp']
    subprocess.run(['git','merge-base','--is-ancestor',commit,launch['commit']],cwd=ROOT,check=True)
    assert record(p['project_state']['path'])==p['project_state']
    assert record(p['selection_protocol']['path'])==p['selection_protocol']
    state=read(ROOT/'project_state.json');assert state['stages']['4']['status']=='FULL_PASS' and state['stages']['5']['status']=='NOT_STARTED'
    assert p['evaluation_decoding']==dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=1024,seed=20260914)


def test_actor_wrapper_only_changes_objective_and_observability():
    import ast
    old=ast.parse((ROOT/'src/medical_posttrain/rl/actor.py').read_text());new=ast.parse((ROOT/'scripts/loss_objective_actor.py').read_text())
    a=next(n for n in old.body if isinstance(n,ast.ClassDef));b=next(n for n in new.body if isinstance(n,ast.ClassDef))
    methods=lambda c:{n.name:n for n in c.body if isinstance(n,ast.FunctionDef)}
    aa,bb=methods(a),methods(b)
    for name in ['logprobs','digest','close']:assert ast.dump(aa[name])==ast.dump(bb[name])
    assert 'compute_policy_loss_gspo' not in (ROOT/'scripts/loss_objective_actor.py').read_text()
    assert (ROOT/'scripts/loss_objective_actor.py').read_text().count('compute_policy_loss_vanilla(')==1
