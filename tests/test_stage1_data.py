import itertools
from pathlib import Path
import pytest
from medical_posttrain.data.format import target
from medical_posttrain.data.stage1 import DuplicateGraph, normalize_question, grams, quality_reason

def test_normalization_preserves_medical_values():
    assert normalize_question('12、患者未服用 1.5mg 药物？') == normalize_question('患者未服用１.５ｍｇ药物?')
    assert normalize_question('服用1.5mg') != normalize_question('服用15mg')
    assert normalize_question('未服用药物') != normalize_question('服用药物')
    assert normalize_question('问题？ A.甲 B.乙') == normalize_question('问题? B.乙 A.甲')
    assert normalize_question('A、B两种药物的安全性') == 'ab两种药物的安全性'

def test_duplicate_join_and_transitive_benchmark_cluster():
    texts=['患者持续高热三天并出现头痛呕吐现象需要检查什么', '患者持续高热三天并出现头痛呕吐现象需要检查什么', '患者持续高热三天并出现头痛呕吐现象需要检查哪些', '全然无关的骨折手术问诊']
    graph=DuplicateGraph(texts); edges=[];graph.join(lambda *args:edges.append(args))
    assert graph.root(0)==graph.root(1)==graph.root(2)
    assert graph.root(3)!=graph.root(0)
    assert len(edges)==2

def test_prefix_join_does_not_lose_high_jaccard_pairs():
    base='患者持续高热三天并出现头痛呕吐现象需要检查什么项目以便确认疾病'
    texts=[base[:i]+'X'+base[i+1:] for i in range(len(base))]
    graph=DuplicateGraph(texts); graph.join(lambda *a:None)
    for i,j in itertools.combinations(range(len(texts)),2):
        a,b=grams(texts[i],5),grams(texts[j],5)
        if len(a&b)/len(a|b)>=.85: assert graph.root(i)==graph.root(j)

def test_obvious_quality_filters():
    assert quality_reason([dict(role='user',content='坏字符\ufffd'),dict(role='assistant',content=target('答案'))])=='encoding_corruption'
    assert quality_reason([dict(role='user',content='病症？'),dict(role='assistant',content=target('检查'))]) is None

@pytest.fixture(scope='module')
def tokenizer():
    transformers=pytest.importorskip('transformers')
    return transformers.AutoTokenizer.from_pretrained('/data/WSH/medical-post-train-artifacts/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218',local_files_only=True)

def test_real_tokenizer_reasoning_and_multiple_assistants(tokenizer):
    from medical_posttrain.data.stage1 import encode_conversation,collate
    messages=[dict(role='system',content='系统提示'),dict(role='user',content='用户一'),dict(role='assistant',content=target('回答一')),dict(role='user',content='用户二'),dict(role='assistant',content=target('回答二','真实推理'))]
    row=encode_conversation(tokenizer,messages)
    supervised=tokenizer.decode([x for x in row['labels'] if x!=-100])
    assert '系统提示' not in supervised and '用户一' not in supervised and '用户二' not in supervised
    assert '回答一' in supervised and '回答二' in supervised and '真实推理' in supervised
    assert supervised.count(tokenizer.eos_token)==2
    assert row['assistant_turns']==2 and row['native_history_think_blocks_removed']==1
    shorter=encode_conversation(tokenizer,[dict(role='user',content='问'),dict(role='assistant',content=target('答'))])
    batch=collate([row,shorter],tokenizer.eos_token_id)
    n=len(shorter['input_ids'])
    assert batch['labels'][1,n:].eq(-100).all()
    assert batch['attention_mask'][1,n:].eq(0).all()
    assert batch['labels'][1,:n].eq(tokenizer.eos_token_id).sum()==1
    assert shorter['reasoning_tokens']==0
