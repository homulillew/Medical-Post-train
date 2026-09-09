"""Fixed CMExam monitor512 only; selection/test remain outside Stage4."""
import csv
from pathlib import Path
import time

from .common import read,record,sha256,immutable


def monitor_rows(protocol):
    from medical_posttrain.data.exam import options_schema,canonical_answer
    source = protocol['source']
    assert sha256(source['path'])==source['sha256']
    assert record(protocol['partition']['path'])==protocol['partition']
    partition = read(protocol['partition']['path'])
    assert protocol['monitor_ids']==partition['monitor'] and len(partition['monitor'])==512
    assert not set(partition['monitor']) & set(partition['selection'])
    raw = list(csv.DictReader(open(source['path'])))
    rows = []
    for pid in partition['monitor']:
        assert pid.startswith('cmexam_val:')
        r = raw[int(pid.split(':')[-1])]
        options = options_schema(r['Options'])
        rows.append(dict(prompt_id=pid,question=r['Question'],options=options,
            answer_set=canonical_answer(r['Answer'],''.join(options)),reference_explanation='',split='val'))
    return rows


def evaluate(rollout,out,state,protocol_ref):
    from vllm import SamplingParams
    from medical_posttrain.reward.parser import parse
    from .actor import array_stats
    out = Path(out)
    directory = out/'validation'/f'{state["policy_windows"]:04d}'
    if (directory/'summary.json').exists():
        summary=read(directory/'summary.json')
        assert summary['policy_version']==state['policy_version']
        assert summary['protocol']==protocol_ref and record(protocol_ref['path'])==protocol_ref
        assert all(summary[k]==state[k] for k in ('policy_windows','optimizer_steps','training_groups'))
        return summary
    directory.mkdir(parents=True,exist_ok=True)
    assert record(protocol_ref['path'])==protocol_ref
    protocol = read(protocol_ref['path'])
    rows = monitor_rows(protocol)
    predictions = []
    seconds = 0.
    for start in range(0,512,16):
        sources = rows[start:start+16]
        path = directory/f'batch_{start//16:03d}.json'
        reservation = directory/f'reservation_{start//16:03d}.json'
        prompts = [rollout.prompt(r) for r in sources]
        if path.exists():
            saved=read(path)
            predictions.extend(saved['predictions'])
            seconds+=saved['seconds']
            continue
        assert not reservation.exists(), 'Interrupted validation has unknown tail cost; retain and investigate'
        immutable(reservation,dict(prompt_ids=[r['prompt_id'] for r in sources],
            prompt_token_ids=[p[1] for p in prompts],policy_version=rollout.policy))
        params = SamplingParams(**protocol['decoding'],seed=protocol['seed'])
        began = time.monotonic()
        results=rollout.llm.generate([p[0] for p in prompts],params,lora_request=rollout.request,use_tqdm=False)
        elapsed=time.monotonic()-began
        assert len(results)==len(sources)==16, 'Incomplete validation batch'
        batch=[]
        for r,p,result in zip(sources,prompts,results):
            assert result.prompt_token_ids==p[1] and len(result.outputs)==1
            o=result.outputs[0]
            parsed=parse(o.text,''.join(r['options']),o.finish_reason)
            batch.append(dict(prompt_id=r['prompt_id'],prompt_token_ids=p[1],response_ids=list(o.token_ids),
                output=o.text,finish_reason=o.finish_reason,policy_version=rollout.policy,
                answer_set=r['answer_set'],parsed_answer=parsed.answer_set,
                acc=int(parsed.answer_set==r['answer_set']),strict_format=parsed.valid_format,
                answer_closure='<answer>' in o.text and '</answer>' in o.text,
                think_closure='<think>' in o.text and '</think>' in o.text))
        immutable(path,dict(predictions=batch,seconds=elapsed))
        predictions.extend(batch)
        seconds+=elapsed
    assert [p['prompt_id'] for p in predictions]==protocol['monitor_ids']
    summary=dict(policy_version=state['policy_version'],policy_windows=state['policy_windows'],
        optimizer_steps=state['optimizer_steps'],training_groups=state['training_groups'],n=512,
        accuracy=sum(r['acc'] for r in predictions)/512,
        strict_format=sum(r['strict_format'] for r in predictions)/512,
        answer_closure=sum(r['answer_closure'] for r in predictions)/512,
        think_closure=sum(r['think_closure'] for r in predictions)/512,
        response_length=array_stats([len(r['response_ids']) for r in predictions]),
        truncation=sum(r['finish_reason']=='length' for r in predictions)/512,
        cumulative_generated_prompt_tokens=state['prompt_tokens'],
        cumulative_generated_logical_prompt_tokens=4*state['prompt_tokens'],
        cumulative_generated_output_tokens=state['output_tokens'],
        cumulative_generated_total_tokens=state['prompt_tokens']+state['output_tokens'],
        validation_output_tokens=sum(len(r['response_ids']) for r in predictions),
        validation_prompt_tokens=sum(len(r['prompt_token_ids']) for r in predictions),
        seconds=seconds,protocol=protocol_ref,test_used=False,selection_used=False)
    immutable(directory/'summary.json',summary)
    return summary
