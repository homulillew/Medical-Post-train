#!/usr/bin/env python3
"""Create an evidence handoff; narrative authorship and scientific gates stay separate."""
import argparse
import datetime
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now
from frontier_diagnostic import gates,INDEX as FRONTIER
INDEX=ROOT/'experiments/stage4'


def build():
    pair=gates()
    analysis=read(INDEX/'postformal_analysis_v1.json')
    last=analysis['accuracy_vs_update_budget'][-1]
    freeze=read(ROOT/'experiments/stage5/dataset_freeze_v1.json')
    # Metadata/hash inspection only: no held-out inference or re-selection.
    for ref in freeze['dataset_manifests']:
        assert record(ref['path'])==ref
    assert record(freeze['verification']['path'])==freeze['verification']
    frontier_done=(FRONTIER/'verification.json').exists()
    if frontier_done:assert read(FRONTIER/'verification.json')['result']=='PASS'
    def evidence(p):return record(p) if Path(p).exists() else None
    pending=['ChatGPT-authored final stage narrative and interview story',
             'Stage4 full acceptance verifier after narrative/manual-review deliverables are available',
             'Stage5 validation-only deployment-checkpoint selection protocol and execution; selection1024 unconsumed',
             'Stage5 project-model final test/open-QA inference and statistical/manual audits',
             'Complete truncated owner PhaseD instruction before final Random-3x protocol freeze']
    if not frontier_done:pending.insert(0,'Finish all12000 Frontier responses and raw replay/bootstrap; launched work is incomplete')
    data=dict(timestamp=now(),kind='MACHINE_READABLE_HANDOFF',narrative_author='ChatGPT/user',
        stage4_done=False,stage5_ready=False,stage5_status=read(ROOT/'project_state.json')['stages']['5']['status'],
        formal_pair=record(INDEX/'formal_pair.json'),raw_receipts={n:record(INDEX/f'{n}_formal_verification.json') for n in pair['runs']},
        final_native_reloads={n:record(Path(r['path'])/'final_reload/result.json') for n,r in pair['runs'].items()},
        analysis=record(INDEX/'postformal_analysis_v1.json'),plot_data=record(INDEX/'postformal_data_v1/manifest.json'),
        existing_full_analysis=record(INDEX/'formal_analysis_v1.json'),
        checkpoint_inventory=record(INDEX/'formal_checkpoint_index.json'),
        supported_observations=dict(
            full_budget_reached={'vanilla':5000,'dynamic':5000},
            equal_update_deltas_pp=[dict(groups=r['training_groups'],delta=100*r['paired']['accuracy_delta']) for r in analysis['accuracy_vs_update_budget']],
            maximum_observed_milestone_delta=analysis['observations'],
            final_difference=dict(delta_pp=100*last['paired']['accuracy_delta'],corrected=last['paired']['corrected'],regressed=last['paired']['regressed'],
                exact_p_exploratory=last['paired']['exact_p_unadjusted'],paired_bootstrap=last['bootstrap']),
            costs=analysis['cost_decomposition'],
            repeated_generated_prompts={n:r['exposure']['generated']['repeated_prompt_count'] for n,r in analysis['runs'].items()},
            health={n:{k:r['optimization_health'][k] for k in ['first_mini_clips_all_zero','max_consecutive_clip_gt_09','optimizer_inactive_windows','nonfinite_values']} for n,r in analysis['runs'].items()}),
        unsupported_claims=['Statistically credible final monitor superiority','Universal superiority across training seeds',
            'Lower rollout cost or stable equal-token advantage','Token ratio equals GPU-hour ratio',
            'Observed advantage is confined to early/mid training: maximum measured group-grid delta is at4096',
            'Training-population shifts prove individual prompts crossed a learned frontier'],
        hypotheses_to_test=['Frontier mixed resolution H1','Mixed-to-all-wrong regression H2','Parsed-wrong vs parser-only resolution H3','Hard all-wrong and mixed mass shifts H4'],
        frontier=dict(status=read(FRONTIER/'status.json'),preregistration=record(FRONTIER/'preregistration.json'),manifest=record(FRONTIER/'manifest.json'),
            analysis=evidence(FRONTIER/'analysis.json'),verification=evidence(FRONTIER/'verification.json')),
        case_evidence=dict(automatic_formal_cases='formal_analysis_v1.json:runs.*.automatic_cases',
            raw_paired_disagreements='postformal_analysis_v1.json:accuracy_vs_update_budget[*].paired.corrected_ids/regressed_ids',
            historical_system_failure=record(ROOT/'docs/implementation/STAGE4_GPU_RELEASE_RECOVERY.md'),
            human_manual_review='Not backfilled; machine-mined cases are not human-reviewed cases'),
        stage5_preparation=dict(dataset_freeze=record(ROOT/'experiments/stage5/dataset_freeze_v1.json'),
            dataset_hash_checks='PASS',selection1024_inference_performed=False,project_model_test_inference_performed=False,
            final_endpoints_for_frontier='formal_checkpoint_index.json:*:final; never deployment-selected endpoints',
            selection_policy='Must be separately frozen before selection1024 inference; frontier outputs may not choose checkpoints',
            external_paid_judge_status='STOPPED_BY_OWNER_BUDGET; no new API requests',
            unresolved_manual_audit='Open-ended scoring/manual audits remain required; no clinician verification claim'),
        pending=pending,
        artifacts_retention='Git stores machine evidence, hashes, plots and selected cases. Full raw output and checkpoint payloads remain at manifest artifact paths.')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    file=INDEX/f'postformal_handoff_{stamp}.json'
    immutable(file,data)
    durable(INDEX/'postformal_handoff_current.json',dict(handoff=record(file),timestamp=now(),frontier_complete=frontier_done,stage4_done=False))
    print(file)


if __name__=='__main__':build()
