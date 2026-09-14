#!/usr/bin/env python3
"""Offline documentation checks. Never invokes a stage verifier or model runtime."""
import datetime
import json
import math
from pathlib import Path
import subprocess

from checkpoint_selection import ROOT, freeze_json, read, reference, require_stage4_gate

S4 = ROOT / 'experiments/stage4'
HANDOFF = ROOT / 'experiments/handoffs/stage4_final_docs_to_chatgpt_v1.json'
PROMPT = Path('/home/ubuntu/.codex/attachments/8bb027d0-142f-4690-b4e4-946fdf307a6b/pasted-text.txt')
REPORT = ROOT / 'docs/stage_reports/04_gspo.md'
STORY = ROOT / 'docs/stage_reports/04_gspo_interview_story.md'
# Exact evidence values and their required report rendering. Rounded displays
# are fixed here before drafting; this is not a recomputation of experiments.
CLAIMS = [
    ('formal_pair/equal_update_curve/0/vanilla/accuracy', .55078125, '55.078125%'),
    ('formal_pair/final_metrics/vanilla/accuracy', .67578125, '67.578125%'),
    ('formal_pair/final_metrics/dynamic/accuracy', .685546875, '68.5546875%'),
    ('formal_pair/final_metrics/paired/corrected', 44, '44'),
    ('formal_pair/final_metrics/paired/regressed', 39, '39'),
    ('formal_pair/final_metrics/paired/exact_p_unadjusted', .6608836477612148, '0.660884'),
    ('formal_pair/final_metrics/bootstrap/ci95/0', -.025390625, '−2.5390625'),
    ('formal_pair/final_metrics/bootstrap/ci95/1', .044921875, '+4.4921875'),
    ('formal_pair/observations/final_delta_pp', .9765625, '+0.9765625'),
    ('formal_pair/observations/largest_observed_group_milestone_delta_pp', 4.1015625, '+4.1015625'),
    ('formal_pair/cost_accounting/group_ratio', 2.9744, '2.9744'),
    ('formal_pair/cost_accounting/total_rollout_token_ratio', 2.924538241234187, '2.92454'),
    ('formal_pair/final_metrics/vanilla/cumulative_generated_total_tokens', 5964478, '5,964,478'),
    ('formal_pair/final_metrics/dynamic/cumulative_generated_total_tokens', 17443344, '17,443,344'),
    ('formal_pair/final_metrics/dynamic/generated_groups', 14872, '14,872'),
    ('signal_density/runs/vanilla/densities/correctness_contrast', .4102, '41.02%'),
    ('signal_density/runs/vanilla/densities/correctness_discriminative', .4102, '41.02%'),
    ('signal_density/runs/vanilla/densities/nonzero_advantage', .757, '75.7%'),
    ('signal_density/runs/dynamic/densities/correctness_discriminative', 1., '100%'),
    ('signal_density/runs/dynamic/densities/nonzero_advantage', 1., '100%'),
    ('signal_density/runs/vanilla/by_type/all_correct/nonzero_group_fraction', .8464606181455633, '84.65%'),
    ('signal_density/effective_signal_amplification', 2.4378352023403216, '2.4378'),
    ('frontier/results/policy_summaries/vanilla/accuracy', .673, '67.3%'),
    ('frontier/results/policy_summaries/dynamic/accuracy', .686, '68.6%'),
    ('frontier/results/sft_subgroups/mixed/all_correct/n', 517, '517'),
    ('frontier/results/sft_subgroups/mixed/all_correct/a_mean', 264/517, '51.06%'),
    ('frontier/results/sft_subgroups/mixed/all_correct/b_mean', 261/517, '50.48%'),
    ('frontier/results/sft_subgroups/all_wrong/accuracy/n', 231, '231'),
    ('frontier/results/sft_subgroups/all_wrong/accuracy/a_mean', 223/924, '24.13%'),
    ('frontier/results/sft_subgroups/all_wrong/accuracy/b_mean', 271/924, '29.33%'),
    ('frontier/results/sft_subgroups/all_wrong/accuracy/ci95/0', .021645021645021644, '+2.16'),
    ('frontier/results/sft_subgroups/all_wrong/accuracy/ci95/1', .08333333333333333, '+8.33'),
    ('random3x/training_comparison/runs/vanilla/signal/densities/correctness_discriminative', .482421875, '48.2421875%'),
    ('random3x/training_comparison/runs/random3x/signal/densities/correctness_discriminative', .4453125, '44.53125%'),
    ('random3x/training_comparison/runs/dynamic/generated_groups', 1360, '1,360'),
    ('random3x/training_comparison/runs/random3x/generated_groups', 1360, '1,360'),
    ('random3x/training_comparison/runs/dynamic/rollout_tokens', 1633072, '1,633,072'),
    ('random3x/training_comparison/runs/random3x/rollout_tokens', 1627034, '1,627,034'),
    ('random3x/training_comparison/runs/dynamic/active_phase_hours', 3.027608377941263, '3.03'),
    ('random3x/training_comparison/runs/random3x/active_phase_hours', 3.0605719615314673, '3.06'),
    ('random3x/evaluation/summary/sft/accuracy', .599609375, '59.9609375%'),
    ('random3x/evaluation/summary/vanilla/accuracy', .638671875, '63.8671875%'),
    ('random3x/evaluation/summary/random3x/accuracy', .625, '62.5%'),
    ('random3x/evaluation/summary/dynamic/accuracy', .646484375, '64.6484375%'),
    ('random3x/evaluation/paired/dynamic_minus_random3x/wrong_to_correct', 43, '43'),
    ('random3x/evaluation/paired/dynamic_minus_random3x/correct_to_wrong', 32, '32'),
    ('random3x/evaluation/paired/dynamic_minus_random3x/exact_mcnemar', .24804574189252795, '0.248046'),
    ('random3x/evaluation/paired/dynamic_minus_random3x/bootstrap/delta', .021484375, '+2.1484375'),
    ('random3x/evaluation/paired/dynamic_minus_random3x/bootstrap/ci95/0', -.01171875, '−1.171875'),
    ('random3x/evaluation/paired/dynamic_minus_random3x/bootstrap/ci95/1', .0546875, '+5.46875'),
    ('random3x/evaluation/paired/random3x_minus_vanilla/bootstrap/delta', -.013671875, '−1.3671875'),
]


def at(value, pointer):
    for key in pointer.split('/'):
        value = value[int(key)] if isinstance(value,list) else value[key]
    return value


def preflight():
    try:
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        remote = subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
        assert head == remote == '8a5d137739bf2246c202ce05d6b112c23b02c1a2'
        state = read(ROOT/'project_state.json')
        assert {k:v['status'] for k,v in state['stages'].items()} == {
            '1':'DONE','2':'DONE','3':'DONE','4':'FULL_PASS','5':'NOT_STARTED','6':'NOT_STARTED'}
        closure = read(S4/'closure_verification_v1.json'); assert closure['result']=='PASS'
        for source in closure['artifacts']: assert reference(source['path'])==source
        h = read(HANDOFF)
        for pointer,expected,_ in CLAIMS:
            assert math.isclose(at(h,pointer),expected,rel_tol=0,abs_tol=1e-12), pointer
        assert h['random3x']['mechanism_status']=='INCONCLUSIVE'
        assert h['selection1024_untouched'] and h['final_tests_untouched']
        assert all(p['dynamic']['accuracy']>p['vanilla']['accuracy'] for p in h['formal_pair']['equal_update_curve'] if p['training_groups']>0)
        assert h['formal_pair']['observations']['dynamic_wins_shared_token_thresholds']==[4000000]
        assert h['formal_pair']['observations']['largest_delta_group_milestones']==[4096]
        for v in ['vanilla','dynamic']:
            health=h['formal_pair']['run_statistics'][v]['optimization_health']
            assert health['nonfinite_values']==health['optimizer_inactive_windows']==0
        cases={c['case_id']:c for c in read(S4/'manual_review_packet_v1.json')['cases']}
        for cid,answers in [('S4-MR-02',['ACE','E','A','C']),('S4-MR-03',['B','B','C','C'])]:
            c=cases[cid]
            assert not c['human_reviewed'] and not c['read_in_full']
            for model,answer in zip(['sft','vanilla','dynamic','random3x'],answers):
                r=c['models'][model]['responses'][0]
                assert r['parsed']['answer_set']==answer and r['parsed']['valid_format']
                assert reference(r['raw_source']['path'])==r['raw_source']
        assert not (S4/'deliverables.json').exists()
        assert not (S4/'formal_manual_cases_v1.json').exists()
        assert not read(ROOT/'experiments/stage5/checkpoint_selection_protocol_v1.json')['generation_performed']
        try: require_stage4_gate()
        except PermissionError: pass
        else: raise AssertionError('Selection gate unexpectedly open')
        files=subprocess.check_output(['git','ls-files','experiments','src','configs','contracts','scripts','tests','project_state.json'],cwd=ROOT,text=True).splitlines()
        protected=[reference(ROOT/f) for f in files if (ROOT/f).is_file()]
        freeze_json(S4/'documentation_input_audit_v1.json',dict(result='PASS',scope='USER_NARRATIVE_FACT_CHECK_ONLY',
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),HEAD=head,remote_main=remote,
            owner_instruction=reference(PROMPT),handoff=reference(HANDOFF),project_state=reference(ROOT/'project_state.json'),
            protected_sources=protected,README_before=reference(ROOT/'README.md'),
            claims=[dict(pointer=p,expected=e,report_literal=t) for p,e,t in CLAIMS],
            conflicts=[],manual_review_count=0,manual_review_scope='Final Stage4 reviews; earlier pilot reviews are separate.',
            narrative_owner='ChatGPT text supplied by user; Codex wires files and checks evidence',
            full_verifier_run=False,READY_FOR_STAGE5='NO'))
    except Exception as exc:
        path=S4/f'documentation_conflict_{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.json'
        freeze_json(path,dict(status='CONFLICT_STOP_FINAL_WIRING',error=repr(exc),owner_instruction=reference(PROMPT)))
        raise


def verify():
    audit=read(S4/'documentation_input_audit_v1.json')
    for source in audit['protected_sources']: assert reference(source['path'])==source
    h=read(HANDOFF); report=REPORT.read_text(); story=STORY.read_text()
    for claim in audit['claims']:
        assert math.isclose(at(h,claim['pointer']),claim['expected'],rel_tol=0,abs_tol=1e-12)
        assert claim['report_literal'] in report, claim
    for variant in ['vanilla','dynamic']:
        assert h['formal_pair']['runs'][variant]['run_id'] in report
    for phrase in ['GRPO-style','GSPO','sequence','INCONCLUSIVE']:
        assert phrase in report and phrase in story
    draft=read(S4/'formal_manual_cases_draft_v1.json')
    assert draft['status']=='WAITING_FOR_USER_CONFIRMATION' and draft['manual_reviews']==[]
    assert {r['case_id'] for r in draft['proposed_reviews']}=={'S4-MR-02','S4-MR-03'}
    for review in draft['proposed_reviews']:
        assert review['read_in_full'] is review['user_confirmed'] is review['human_reviewed'] is False
        for source in review['sources']: assert reference(source['path'])==source
    manifest=read(S4/'deliverables_draft_v1.json')
    assert manifest['READY_FOR_STAGE5']=='NO'
    for key in ['stage_report','interview_story','cases','resume_evidence','plots','checkpoint_index','pilot_review']:
        refs=manifest[key] if isinstance(manifest[key],list) else [manifest[key]]
        assert refs
        for source in refs: assert reference(source['path'])==source
    assert not (S4/'deliverables.json').exists()
    assert reference(ROOT/'project_state.json')==audit['project_state']
    try: require_stage4_gate()
    except PermissionError: pass
    else: raise AssertionError('Gate opened without genuine review')
    return dict(result='PASS',scope='DOCUMENTATION_DRAFT_CONSISTENCY_ONLY',
                numeric_claims_checked=len(audit['claims']),protected_sources_unchanged=len(audit['protected_sources']),
                human_review_count=0,full_verifier_run=False,READY_FOR_STAGE5='NO')


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['preflight']: preflight()
    else: print(json.dumps(verify(),indent=2))
