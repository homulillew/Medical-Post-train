from copy import deepcopy
from types import SimpleNamespace
import pytest

def test_token_weighted_accumulation_matches_padded_batch():
    torch=pytest.importorskip('torch')
    from medical_posttrain.training.stage1 import update
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__();self.embedding=torch.nn.Embedding(16,4);self.head=torch.nn.Linear(4,16)
        def forward(self,input_ids,labels,attention_mask):
            logits=self.head(self.embedding(input_ids))[:,:-1]
            return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits.reshape(-1,16),labels[:,1:].reshape(-1),ignore_index=-100))
    torch.manual_seed(42);base=TinyModel()
    rows=[dict(sample_id=str(i),input_ids=[2]*i+[3,4,0],labels=[-100]*i+[3,4,0],total_tokens=i+3) for i in (1,4,2,6)]
    outputs=[]
    for microbatch in (1,2,4):
        model=deepcopy(base);optimizer=torch.optim.SGD(model.parameters(),lr=.1);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:1.)
        result=update(model,SimpleNamespace(pad_token_id=0),optimizer,scheduler,rows,dict(microbatch=microbatch,max_grad_norm=1.,sort_within_effective_batch=True))
        outputs.append((model,result))
    for model,result in outputs[1:]:
        assert abs(result['loss']-outputs[0][1]['loss'])<1e-6
        for a,b in zip(model.parameters(),outputs[0][0].parameters()):assert torch.allclose(a,b,atol=1e-7,rtol=1e-6)
        assert result['sample_ids']==['1','4','2','6'] and result['supervised_tokens']==12
