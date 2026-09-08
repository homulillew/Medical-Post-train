"""Strict source-to-conversation adapters for the forthcoming Stage 1 smoke."""
from medical_posttrain.data.format import target

def medical_o1(row):
    required={'Question','Complex_CoT','Response'}
    if set(row)!=required or not all(isinstance(row[k],str) for k in required):
        raise ValueError('Expected the pinned Chinese medical-o1 fields')
    if not row['Question'].strip() or not row['Response'].strip():
        raise ValueError('Empty medical-o1 question/response')
    return [{'role':'user','content':row['Question']},{'role':'assistant','content':target(row['Response'],row['Complex_CoT'])}]

def huatuo(row):
    if set(row)!={'id','conversations'} or not isinstance(row['conversations'],list):
        raise ValueError('Expected pinned Huatuo id/conversations schema')
    result=[]
    for i,turn in enumerate(row['conversations']):
        if set(turn)!={'from','value'} or turn['from']!=('human' if i%2==0 else 'gpt') or not isinstance(turn['value'],str) or not turn['value'].strip():
            raise ValueError('Malformed or non-alternating Huatuo conversation')
        result.append({'role':'user' if i%2==0 else 'assistant','content':turn['value'] if i%2==0 else target(turn['value'])})
    if not result or len(result)%2:raise ValueError('Huatuo conversation must end with assistant')
    return result
