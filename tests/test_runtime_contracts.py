import json
from pathlib import Path
import pytest
from medical_posttrain.config import ProbeConfig,read_config
from medical_posttrain.data.format import parse_answer,target

def test_strict_config(tmp_path):
    p=tmp_path/'config.json'; p.write_text('{"purpose":"lora","silent_fallback":true}')
    with pytest.raises(TypeError):read_config(p)
    with pytest.raises(ValueError):ProbeConfig('lora',stage=1)
    with pytest.raises(ValueError):ProbeConfig('lora',run_class='FORMAL')
    with pytest.raises(ValueError):ProbeConfig('unknown')

@pytest.mark.parametrize('text,answer',[
    ('<answer>C</answer>','C'),('<answer> B、A </answer>','AB'),
    (target('C','原始推理'),'C'),(target('C'),'C'),
    ('<answer>CC</answer>',None),('<answer>F</answer>',None),
    ('<answer>C',None),('<answer>C</answer><answer>A</answer>',None),
    ('<think><answer>A</answer></think><answer>C</answer>',None),
    ('<answer>C</answer>其他文字',None),
    ('<think></think><think></think><answer>C</answer>',None),
])
def test_parser_ambiguity(text,answer):assert parse_answer(text)==answer

def test_no_invented_huatuo_reasoning():
    assert target('原始回答')=='<think>\n\n</think>\n\n<answer>原始回答</answer>'
    with pytest.raises(ValueError):target('<answer>原始</answer>')
