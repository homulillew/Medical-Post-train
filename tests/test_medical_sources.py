import pytest
from medical_posttrain.data.medical import medical_o1,huatuo

def test_sources_preserve_available_reasoning():
    o=medical_o1({'Question':'问题','Complex_CoT':'原始推理','Response':'答案'})
    assert '原始推理' in o[-1]['content']
    h=huatuo({'id':'fixture','conversations':[{'from':'human','value':'问题'},{'from':'gpt','value':'原始回答'}]})
    assert h[-1]['content']=='<think>\n\n</think>\n\n<answer>原始回答</answer>'

def test_invalid_source_fails_explicitly():
    with pytest.raises(ValueError):medical_o1({'Question':'问题','Response':'答案'})
    with pytest.raises(ValueError):huatuo({'id':'fixture','conversations':[{'from':'gpt','value':'回答'}]})
