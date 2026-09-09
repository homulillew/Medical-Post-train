"""Stage 2 entry point: dataset, diagnostics, fixed-policy rollout and scoring only."""
import argparse
from pathlib import Path
import traceback
from medical_posttrain.evidence import write_json, now, sha256
from medical_posttrain.evidence.stage2 import prepare, launch, Run, read, select, selected_path, INDEX


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['data','semantic','rollout','score','worker','resume']);p.add_argument('--run');p.add_argument('--action');p.add_argument('--attempt');p.add_argument('--config');p.add_argument('--mode',default='smoke',choices=['smoke','formal','length']);a=p.parse_args()
    if a.command=='worker':
        run=Run(a.run,a.attempt)
        try:
            if a.action=='data':
                from medical_posttrain.data.exam import prepare_pool
                prepare_pool(run)
            elif a.action=='semantic':
                from medical_posttrain.reward.semantic import diagnostic
                diagnostic(run)
            elif a.action=='rollout':
                from medical_posttrain.rollout.profile import rollout
                rollout(run)
            elif a.action=='score':
                from medical_posttrain.rollout.profile import score_run
                score_run(run)
            else:raise ValueError(a.action)
        except BaseException:
            failure=dict(status='FAILED',ended=now(),traceback=traceback.format_exc())
            write_json(run.attempt/'status.json',failure);write_json(run.out/'status.json',failure);raise
        return
    if a.command=='resume':
        out=Path(a.run)
        assert read(out/'status.json')['status'] in ('FAILED','INTERRUPTED')
        launch(out,a.action or 'rollout');return
    if a.command=='score':
        launch(Path(a.run),'score');return
    if a.command=='data':
        state=read('project_state.json');assert state['stage0']['status']=='VERIFIED' and state['stages']['1']['status']=='DONE'
        for s in ('3','4','5','6'):assert state['stages'][s]['status']=='NOT_STARTED'
        data=read('experiments/stage1/selected_runs.json')['data']
        m=read(Path('experiments/stage1')/data/'manifest.json');clusters=Path(m['artifact_root'])/'clusters.jsonl'
        cfg=dict(seed=42,raw_manifest=str(Path('experiments/stage1')/data/'raw_manifest.json'),clusters=str(clusters),clusters_sha256=sha256(clusters),near_rules=m['near_rules'])
        out=prepare('data','DIAGNOSTIC',cfg);select('data',out);launch(out,'data')
    elif a.command=='semantic':
        cfg=dict(seed=42,pool=str(selected_path('data')/'candidate_pool.jsonl'),controlled_pairs=200,device='cpu',chunk_tokens=480)
        out=prepare('semantic','DIAGNOSTIC',cfg);select('semantic',out);launch(out,'semantic')
    elif a.command=='rollout':
        assert a.config
        cfg=read(a.config);out=prepare(a.mode,'FORMAL' if a.mode=='formal' else 'SMOKE' if a.mode=='smoke' else 'DIAGNOSTIC',cfg)
        select(a.mode,out);launch(out,'rollout')

if __name__=='__main__':main()
