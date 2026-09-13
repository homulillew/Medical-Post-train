#!/usr/bin/env python3
"""Machine-only causal-ablation handoff, candidate cases and publication."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now
from random3x_runtime import check
from analyze_signal_density import inspect_run


def main(out):
    p=check();idx=ROOT/'experiments/stage5';s4=ROOT/'experiments/stage4'
    evaluation=read(idx/'aux_random3x_eval_analysis_v1.json')
    assert read(idx/'aux_random3x_verification_v1.json')['result']=='PASS'
    assert read(idx/'aux_random3x_eval_verification_v1.json')['result']=='PASS'
    formal=read(s4/'signal_density_analysis_v1.json');pair=read(s4/'formal_pair.json')
    signal=inspect_run(out,64)
    immutable(idx/'aux_random3x_signal_analysis_v1.json',dict(timestamp=now(),summary=signal['summary'],rows=signal['rows'],
        sources=signal['sources'],checks=signal['checks'],epsilon=1e-6,gradient_norm_by_group_type='NOT_IDENTIFIABLE_FROM_RETAINED_ARTIFACTS'))
    comparisons={}
    paths={n:Path(r['path']) for n,r in pair['runs'].items()};paths['random3x']=out
    for name,path in paths.items():
        commits=[read(path/'windows'/f'{i:04d}'/'commit.json') for i in range(64)]
        state=commits[-1]['state_after'];minis=[m for i in range(64) for m in read(path/'windows'/f'{i:04d}'/'update/update.json')['minibatches']]
        durations={key:sum(c['metrics'][key] for c in commits) for key in ['actor_process_seconds','sleep_seconds','wake_seconds','sync_seconds']}
        durations['generation_seconds']=sum(read(f)['generation_seconds'] for i in range(64) for f in (path/'windows'/f'{i:04d}'/'batches').glob('*/raw.json'))
        durations['reward_seconds']=sum(read(f)['seconds'] for i in range(64) for f in (path/'windows'/f'{i:04d}'/'batches').glob('*/reward_runtime.json'))
        comparisons[name]=dict(training_groups=state['training_groups'],generated_groups=state['generated_groups'],
            output_tokens=state['output_tokens'],physical_prompt_tokens=state['prompt_tokens'],rollout_tokens=state['output_tokens']+state['prompt_tokens'],
            signal=signal['summary'] if name=='random3x' else formal['runs'][name]['first512'],
            optimization=dict(clip_mean=float(np.mean([m['clip_fraction'] for m in minis])),second_mini_clip_mean=float(np.mean([m['clip_fraction'] for m in minis[1::2]])),
                entropy_mean=float(np.mean([m['entropy'] for m in minis])),grad_norm_mean=float(np.mean([m['grad_norm'] for m in minis])),grad_norm_max=max(m['grad_norm'] for m in minis)),
            generated_response_length_mean=state['output_tokens']/(4*state['generated_groups']),
            active_phase_seconds=durations,active_phase_hours=sum(durations.values())/3600,
            wall_first_to_last_commit_seconds=(path/'windows/0063/commit.json').stat().st_mtime-(path/'windows/0000/state_before.json').stat().st_mtime,
            accounting='Wall span includes any downtime; phase accounting omits outer loading/verification. Actor nested optimizer durations must not be added twice.')
    immutable(idx/'aux_random3x_training_comparison_v1.json',dict(timestamp=now(),runs=comparisons,
        scope='AUXILIARY_SAME512_UPDATE_BUDGET',generation_schedule_matched=True,exact_token_matched=False))
    cases=[]
    def eval_sources(pid,names):
        refs=[]
        for name in names:
            for f in (out/'ablation_eval'/name).glob('batch_*.json'):
                if 'reservation' in f.name:continue
                if any(r['prompt_id']==pid for r in read(f)['predictions']):refs.append(record(f));break
        return refs
    for comparison,names in [('dynamic_minus_vanilla',['vanilla','dynamic']),('dynamic_minus_random3x',['random3x','dynamic']),('random3x_minus_vanilla',['vanilla','random3x'])]:
        for direction,key in [('improvement','gained_ids'),('regression','regressed_ids')]:
            for pid in evaluation['paired'][comparison][key][:2]:
                cases.append(dict(category=comparison+'_'+direction,prompt_id=pid,sources=eval_sources(pid,names),human_reviewed=False))
    frontier=s4/'frontier_diagnostic_v1';fp=read(frontier/'preregistration.json')
    metrics={n:{r['prompt_id']:r for r in read(frontier/f'{n}_prompt_metrics.json')} for n in ['sft','vanilla','dynamic']}
    from medical_posttrain.sampling.dynamic import Stream
    front_rows=read(read(frontier/'manifest.json')['dataset']['path'])
    order=Stream([r['prompt_id'] for r in front_rows],fp['seed'],fp['stream_domain']).order(0)
    filters={
        'sft_all_wrong_dynamic_improvement':lambda s,v,d:s['all_wrong'] and d['correct_count']>s['correct_count'],
        'sft_mixed_dynamic_not_better':lambda s,v,d:s['mixed'] and d['correct_count']<=v['correct_count'],
        'parser_only_mixed':lambda s,v,d:s['classification']=='mixed_unparseable_only',
        'parsed_wrong_mixed':lambda s,v,d:s['classification']=='mixed_parsed_wrong'}
    for category,predicate in filters.items():
        ids=[pid for pid,s in metrics['sft'].items() if predicate(s,metrics['vanilla'][pid],metrics['dynamic'][pid])]
        for pid in ids[:2]:
            batch=order.index(pid)//8
            refs=[record(Path(fp['artifact_path'])/n/'batches'/f'{batch:04d}'/'raw.json') for n in metrics]
            cases.append(dict(category=category,prompt_id=pid,sources=refs,human_reviewed=False,
                observed_counts={n:metrics[n][pid]['correct_count'] for n in metrics}))
    targets={'rejected_all_correct','overflow_eligible'};found=set()
    for w in sorted((paths['dynamic']/'windows').iterdir()):
        for b in sorted((w/'batches').iterdir()):
            decisions=read(b/'dispositions.json')
            for dec in decisions:
                kind=dec['disposition']
                if kind in targets-found:
                    cases.append(dict(category=kind,group_id=dec['group_id'],prompt_id=dec['prompt_id'],
                        sources=[record(b/'raw.json'),record(b/'scored.json'),record(b/'dispositions.json')],human_reviewed=False))
                    found.add(kind)
        if found==targets:break
    cases.append(dict(category='transactional_recovery_system_failure',sources=[record(s4/'recovery_fault_injections.json'),
        record(ROOT/'docs/implementation/STAGE4_GPU_RELEASE_RECOVERY.md')],human_reviewed=False))
    immutable(idx/'aux_random3x_case_candidates_v1.json',dict(timestamp=now(),cases=cases,review_status='MACHINE_CANDIDATES_ONLY',
        selection_rule='First2 observed IDs per prespecified category; illustrative not prevalence estimates'))
    ci=evaluation['paired']['dynamic_minus_random3x']['bootstrap']['ci95']
    conclusion='SUPPORTED' if ci[0]>0 else 'NOT_SUPPORTED' if ci[1]<=0 else 'INCONCLUSIVE'
    handoff=dict(timestamp=now(),current_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        stage4_state=read(ROOT/'project_state.json')['stages']['4']['status'],stage5_state='NOT_STARTED',READY_FOR_STAGE5='NO',
        formal_receipts={n:record(s4/f'{n}_formal_verification.json') for n in pair['runs']},
        frontier_analysis=record(frontier/'analysis.json'),frontier_verification=record(frontier/'verification.json'),
        signal_density=record(s4/'signal_density_analysis_v1.json'),signal_density_verifier=record(s4/'signal_density_verification_v1.json'),
        random3x_protocol=record(ROOT/'experiments/stage5/aux_random3x_protocol_v2.json'),
        random3x_run=dict(run_id=out.name,artifact_root=str(out),config=record(out/'config.json'),summary=record(out/'summary.json')),
        random3x_verifier=record(idx/'aux_random3x_verification_v1.json'),resume=record(idx/'aux_random3x_resume_v1.json'),
        ablation_eval_manifest=p['evaluation_manifest'],ablation_eval_analysis=record(idx/'aux_random3x_eval_analysis_v1.json'),
        headline_numbers=evaluation['summary'],paired_statistics=evaluation['paired'],
        training_comparison=record(idx/'aux_random3x_training_comparison_v1.json'),cases=record(idx/'aux_random3x_case_candidates_v1.json'),
        correctness_aware_selection_mechanism=conclusion,conclusion_rule=p['interpretation_rule'],
        supported_claims=['Filtering changes selected correctness-discriminative advantage density, measured directly from retained native tensors.',
                          'Auxiliary point estimates and paired intervals are reported without retraining or selection.'],
        unsupported_claims=['Universal gradient quality/generalization superiority','Nonsignificance establishes equivalence','Exact token-matched ablation','Final-test superiority'],
        previous_stage4_results_unchanged=True,interpretation_update='Auxiliary validation contrast adds evidence; formal and Frontier findings remain recorded independently.',
        acceptance_blockers=record(s4/'stage4_acceptance_readiness_v1.json'),
        selection1024_untouched=True,final_tests_untouched=True,paid_API_new_calls=0,optimizer_replay=False,
        narrative_author='ChatGPT/user',human_review_fabricated=False,
        next_recommended_action='ChatGPT/user completes genuine manual case reviews and narrative deliverables from this handoff, then run full Stage4 acceptance verifier.')
    path=ROOT/'experiments/handoffs/stage4_causal_ablation_to_chatgpt_v1.json';immutable(path,handoff)
    assert read(ROOT/'project_state.json')==p['project_state_snapshot']
    durable(idx/'aux_random3x_status.json',dict(status='AUXILIARY_VERIFIED_AND_ANALYZED',run_id=out.name,timestamp=now(),READY_FOR_STAGE5='NO'))
    # Publish only this run's output namespace; never absorb another session's staged changes.
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode==0
    files=[f for f in idx.glob('aux_random3x*') if f.is_file()]+[path]
    files+=list((s4/out.name).glob('*.json'))
    for f in files:
        if f.suffix=='.json':read(f)
        assert f.stat().st_size<50*1024*1024
    subprocess.run(['git','add','--',*[str(f.relative_to(ROOT)) for f in files]],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','handoff: retain verified random-control training and causal-ablation evidence'],cwd=ROOT,check=True)
    pushed=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True,text=True)
    durable(idx/'aux_random3x_publication.json',dict(timestamp=now(),result='PUSHED' if pushed.returncode==0 else 'PUSH_FAILED',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),output=pushed.stdout+pushed.stderr))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);main(ap.parse_args().run)
