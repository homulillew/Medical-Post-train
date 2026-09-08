import hashlib
from medical_posttrain.data.format import target,messages,sft_tokens,parse_answer
from medical_posttrain.evidence import write_json

def run(ev):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ev.config['model'],local_files_only=True)
    snapshots={}
    for thinking in (True,False):
        text=tok.apply_chat_template(messages('请选择 C。'),tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
        if thinking: assert text.endswith('<|im_start|>assistant\n')
        else: assert text.endswith('<think>\n\n</think>\n\n')
        snapshots[str(thinking)]=dict(text=text,token_ids=tok.encode(text,add_special_tokens=False))
    for name,reason in [('medical_o1','这是样本自带的推理。'),('huatuo','')]:
        ids,labels=sft_tokens(tok,'请选择 C。','C',reason)
        decoded=tok.decode([i for i in labels if i!=-100])
        assert decoded==target('C',reason)+'<|im_end|>\n',repr(decoded)
        assert parse_answer(target('C',reason))=='C'
        snapshots[name]=dict(text=tok.decode(ids),token_ids=ids,labels=labels)
    for text,expected in [('<answer>C</answer>','C'),('<answer>C, A</answer>','AC'),('<answer>CC</answer>',None),('<answer>F</answer>',None),('<answer>C',None),('<answer>C</answer>说明',None),('<answer>C</answer><answer>A</answer>',None)]:
        assert parse_answer(text)==expected,(text,expected)
    write_json(ev.path/'template_snapshots.json',dict(template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),snapshots=snapshots))
    ev.metric(event='template',fixtures=len(snapshots),parser_cases=7)
    return dict(gates={'template':True},fixtures=len(snapshots))
