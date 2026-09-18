from medical_posttrain.evaluation.core import ref
from medical_posttrain.evaluation.blind import DIMENSIONS
from validate_stage5_review_submission import validate_submission


def fixture(tmp_path):
    p=tmp_path/'raw.txt';p.write_text('Test fixture, not a real judgment.')
    j=dict(pair_id='pair',judge_id='fixture',judge_version='test',judge_is_candidate=False,
        preference='tie',scores={s:{d:2 for d in DIMENSIONS} for s in ['A','B']},
        rationale='Synthetic unit-test value only.',critical_safety_flags=[],raw_judgment=ref(p))
    public=[dict(pair_id='pair')];schedule=dict(entries=[dict(pair_id='pair',prompt_id='item')])
    return j,public,schedule


def test_blank_and_duplicates_fail(tmp_path):
    j,p,s=fixture(tmp_path)
    assert validate_submission([dict(j,preference=None)],[],p,s,[])['status']=='INVALID'
    assert validate_submission([j,j],[],p,s,[])['status']=='INVALID'
    assert validate_submission([],[],p,s,[])['status']=='INCOMPLETE'


def test_critical_flags_extend_real_review_queue(tmp_path):
    j,p,s=fixture(tmp_path)
    r=validate_submission([dict(j,critical_safety_flags=['test flag'])],[],p,s,[])
    assert r['status']=='INCOMPLETE' and r['required_human_items']==1
    assert r['missing_human_pairs']==['pair']


def test_candidate_and_tampered_raw_rejected(tmp_path):
    j,p,s=fixture(tmp_path)
    assert validate_submission([dict(j,judge_is_candidate=True)],[],p,s,[])['status']=='INVALID'
    from pathlib import Path
    Path(j['raw_judgment']['path']).write_text('changed')
    assert validate_submission([j],[],p,s,[])['status']=='INVALID'


def test_empty_human_template_never_counts(tmp_path):
    j,p,s=fixture(tmp_path)
    r=validate_submission([j],[dict(pair_id='pair',reviewed_in_full=False)],p,s,['item'])
    assert r['valid_human_pairs']==0 and r['status']=='INVALID'


def test_schema_pass_is_not_stage_pass(tmp_path):
    j,p,s=fixture(tmp_path)
    r=validate_submission([j],[],p,s,[])
    assert r['status']=='SCHEMA_AND_COVERAGE_PASS'
    assert r['full_stage5_complete'] is False and r['clinical_validation'] is False
