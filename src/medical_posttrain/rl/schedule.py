"""Common first-crossing validation schedule, independent of outcome/variant."""


def validation_trigger(protocol,before,after,target):
    stride=protocol.get('generated_total_token_stride')
    milestones=[]
    if stride:
        assert isinstance(stride,int) and stride>0
        old=before['prompt_tokens']+before['output_tokens']
        new=after['prompt_tokens']+after['output_tokens']
        assert new>=old
        milestones=list(range((old//stride+1)*stride,(new//stride)*stride+1,stride))
    group_due=after['policy_windows'] in protocol['checkpoint_windows'] or after['training_groups']==target
    return dict(group_schedule=group_due,generated_total_token_milestones=milestones,
        evaluate=group_due or bool(milestones),policy_windows=after['policy_windows'],
        actual_generated_total_tokens=after['prompt_tokens']+after['output_tokens'])
