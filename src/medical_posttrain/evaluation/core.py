"""CPU-only I/O, response schema and reproducible paired statistics."""
import hashlib,json,math
from pathlib import Path
from collections import Counter
import numpy as np
from medical_posttrain.reward.parser import parse

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ref(p):
    p=Path(p).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size)
def check_ref(r):
    a=ref(r['path'])
    if any(a[k]!=r[k] for k in a):raise ValueError('Artifact hash/size mismatch: '+r['path'])
    return a

def freeze(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def score(raw,item,checkpoint_id,run_id):
    parsed=parse(raw['raw_output'],''.join(item['options']),raw['finish_reason'])
    return dict(checkpoint_id=checkpoint_id,run_id=run_id,prompt_id=item['id'],
        raw_output=raw['raw_output'],finish_reason=raw['finish_reason'],parsed_answer=parsed.answer_set,ground_truth=item['answer'],
        correct=parsed.answer_set==item['answer'],strict_format=parsed.valid_format,
        parse_method='strict' if parsed.strict_match else 'fallback' if parsed.fallback_match else 'unparseable',
        ambiguous=parsed.ambiguous,unparseable=parsed.answer_set is None,truncated=raw['finish_reason']=='length',
        prompt_tokens=raw['prompt_tokens'],output_tokens=raw['output_tokens'])

def validate_responses(data,ids):
    if [r['prompt_id'] for r in data]!=ids or len(set(ids))!=len(ids):raise ValueError('Missing/duplicate/reordered responses')
    for r in data:
        for k in ['correct','strict_format','ambiguous','unparseable','truncated']:
            if type(r[k]) is not bool:raise ValueError('Boolean schema required: '+k)
        for k in ['prompt_tokens','output_tokens']:
            if type(r[k]) is not int or r[k]<0:raise ValueError('Nonnegative token counts required')
        if r['finish_reason'] not in ['stop','length']:raise ValueError('Technical failure is not a scored response')
        if r['unparseable'] and r['correct']:raise ValueError('Unparseable cannot be correct')
    return True

def summarize(data):
    if not data:raise ValueError('No denominator')
    validate_responses(data,[r['prompt_id'] for r in data]);n=len(data);tokens=np.array([r['output_tokens'] for r in data])
    return dict(n=n,correct_count=sum(r['correct'] for r in data),accuracy=sum(r['correct'] for r in data)/n,
        strict_format_rate=sum(r['strict_format'] for r in data)/n,unparseable_rate=sum(r['unparseable'] for r in data)/n,
        truncation_rate=sum(r['truncated'] for r in data)/n,unparseable_count=sum(r['unparseable'] for r in data),truncation_count=sum(r['truncated'] for r in data),
        mean_output_tokens=float(tokens.mean()),**{f'P{q}_output_tokens':float(np.percentile(tokens,q)) for q in [50,95,99]},
        prompt_tokens=sum(r['prompt_tokens'] for r in data),output_tokens=int(tokens.sum()))

def paired(a,b,seed=20260914,resamples=10000):
    ids=[r['prompt_id'] for r in a]
    if ids!=[r['prompt_id'] for r in b] or len(set(ids))!=len(ids) or not ids:raise ValueError('Paired IDs/order/denominator mismatch')
    d=np.array([int(y['correct'])-int(x['correct']) for x,y in zip(a,b)])
    gains=[i for i,x in zip(ids,d) if x==1];losses=[i for i,x in zip(ids,d) if x==-1];n=len(gains)+len(losses)
    # Sum exact integer binomial coefficients before division; avoids float overflow for6811.
    p=min(1.,(2*sum(math.comb(n,k) for k in range(min(len(gains),len(losses))+1)))/(2**n)) if n else 1.
    rng=np.random.default_rng(seed);draws=[]
    for start in range(0,resamples,100):draws.extend(d[rng.integers(0,len(d),size=(min(100,resamples-start),len(d)))].mean(axis=1))
    return dict(n=len(d),delta_pp=float(d.mean()*100),wrong_to_correct=len(gains),correct_to_wrong=len(losses),gained_ids=gains,regressed_ids=losses,
        exact_mcnemar=p,ci95_pp=(np.percentile(draws,[2.5,97.5])*100).tolist(),seed=seed,resamples=resamples,unit='paired prompt',multiple_comparisons='exploratory unadjusted')

def sliced(data,membership):
    return {name:summarize([r for r in data if r['prompt_id'] in set(ids)]) for name,ids in membership.items() if ids}

def win_tie_loss(judgments):
    counts=Counter(j['preference'] for j in judgments)
    if set(counts)-{'A win','tie','B win'}:raise ValueError('Unknown preference')
    return dict(n=len(judgments),**{k:counts[k] for k in ['A win','tie','B win']})

def wilson(correct,n,z=1.959963984540054):
    if not 0<=correct<=n or n<=0:raise ValueError('Invalid denominator')
    p=correct/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-half,center+half]
