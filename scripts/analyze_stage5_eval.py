#!/usr/bin/env python3
"""Future CPU analysis and blind-bundle assembly from complete retained outputs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze,summarize,paired,sliced
from medical_posttrain.evaluation.blind import build_bundle
IDX=ROOT/'experiments/stage5'

def load_complete(directory,ids,exam=True):
    complete=read(Path(directory)/'complete.json')
    if complete['prompt_ids']!=ids or complete['n']!=len(ids):raise ValueError('Incomplete evaluation denominator/order')
    data=[]
    for r in complete['response_refs']:check_ref(r);data.append(read(r['path']))
    if [r['prompt_id'] for r in data]!=ids:raise ValueError('Raw output identity mismatch')
    if exam and summarize(data)!=complete['summary']:raise ValueError('Summary does not replay')
    return data

def analyze(root,write=True):
    root=Path(root);selection=read(IDX/'selection_result_manifest_v1.json');p=read(IDX/'checkpoint_selection_protocol_v1.json')
    primary={'sft':'sft',**{'selected_'+k:c['checkpoint_id'] for k,c in selection['selected_checkpoints'].items()}}
    scientific={k:c['checkpoint_id'] for k,c in p['scientific_endpoints'].items()};tables={};allraw={};cost={}
    for dataset in ['cmexam_test_scorable','cmb_exam_clean_2000']:
        m=read(IDX/'manifests'/f'{dataset}.json');raw={}
        for name in set(primary.values())|set(scientific.values()):
            d=root/'final'/name.replace(':','_')/dataset;raw[name]=load_complete(d,m['ids']);cost[name+':'+dataset]=dict(prompt_tokens=sum(r['prompt_tokens'] for r in raw[name]),output_tokens=sum(r['output_tokens'] for r in raw[name]))
        allraw[dataset]=raw
        tables[dataset]={}
        for label,models in [('Primary Selected Models',primary),('Scientific Equal-Update Endpoints',scientific)]:
            names=list(models);comparisons={}
            for i,a in enumerate(names):
                for b in names[i+1:]:comparisons[b+'_minus_'+a]=paired(raw[models[a]],raw[models[b]])
            tables[dataset][label]=dict(summary={k:summarize(raw[v]) for k,v in models.items()},paired=comparisons)
    cm=read(IDX/'cmexam_slice_manifest_v1.json');cmb=read(IDX/'cmb_primary_audit_v1.json');med=read(IDX/'cmb_medical_only_manifest_v1.json')
    slices=dict(cmexam={k:dict(difficulty=sliced(data,cm['difficulty']),categories={cat:sliced(data,ids) for cat,ids in cm['categories'].items()}) for k,data in allraw['cmexam_test_scorable'].items()},
        cmb={k:sliced(data,cmb['category_membership']) for k,data in allraw['cmb_exam_clean_2000'].items()})
    secondary={}
    for dataset,ids in [('cmexam_clean',read(IDX/'manifests/cmexam_test_clean.json')['ids']),('cmb_medical_only',med['ids'])]:
        data=allraw['cmexam_test_scorable' if dataset=='cmexam_clean' else 'cmb_exam_clean_2000'];filtered={k:[r for r in v if r['prompt_id'] in set(ids)] for k,v in data.items()}
        secondary[dataset]={}
        for label,models in [('Primary Selected Models',primary),('Scientific Equal-Update Endpoints',scientific)]:
            names=list(models);secondary[dataset][label]=dict(summary={k:summarize(filtered[v]) for k,v in models.items()},paired={b+'_minus_'+a:paired(filtered[models[a]],filtered[models[b]]) for i,a in enumerate(names) for b in names[i+1:]})
    outputs={k:{} for k in primary};items=[]
    for dataset in ['cmb_clin','open_qa_retention_200']:
        m=read(IDX/'manifests'/f'{dataset}.json');items.extend(rows(m['data']['path']))
        for role,name in primary.items():
            data=load_complete(root/'final'/name.replace(':','_')/dataset,m['ids'],False);outputs[role].update({r['prompt_id']:r for r in data})
            cost[name+':'+dataset]=dict(prompt_tokens=sum(r['prompt_tokens'] for r in data),output_tokens=sum(r['output_tokens'] for r in data))
    result=dict(status='EXAM_ANALYZED_OPEN_JUDGE_PENDING',selected_checkpoints=selection['selected_checkpoints'],scientific_endpoints=p['scientific_endpoints'],
        cmexam=tables['cmexam_test_scorable'],cmb_primary=tables['cmb_exam_clean_2000'],cmb_medical_only=secondary['cmb_medical_only'],cmexam_clean_secondary=secondary['cmexam_clean'],
        slices=slices,open_qa='JUDGMENTS_AND_HUMAN_AUDIT_REQUIRED',retention='JUDGMENTS_AND_HUMAN_AUDIT_REQUIRED',safety='FLAG_EVIDENCE_AND_HUMAN_AUDIT_REQUIRED',
        paired_statistics=dict(seed=20260914,resamples=10000),cost=cost,limitations=['Single training seed','Open QA is not clinical validation','Clean CMExam secondary is not full test','CMB medical-only does not replace primary2000'])
    if write:
        freeze(IDX/'project_exam_analysis_v1.json',result)
        build_bundle(items,outputs,read(IDX/'open_qa_blind_schedule_v1.json'),root/'blind_bundle')
    return result
