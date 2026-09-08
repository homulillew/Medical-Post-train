import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stage0_verifier',ROOT/'scripts/verify_stage0.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def test_missing_evidence_fails_closed(tmp_path):
    shutil.copytree(ROOT/'contracts',tmp_path/'contracts')
    shutil.copytree(ROOT/'docs',tmp_path/'docs')
    shutil.copy(ROOT/'project_state.json',tmp_path/'project_state.json')
    result=v.verify(tmp_path)
    assert result['result']=='FAIL' and result['missing_gates']==sorted(v.GATES)

def test_stage_progress_is_not_smoke_completion(tmp_path):
    shutil.copytree(ROOT/'contracts',tmp_path/'contracts')
    shutil.copytree(ROOT/'docs',tmp_path/'docs')
    state=json.loads((ROOT/'project_state.json').read_text())
    state['stages']['1']['status']='SMOKE_PASS'
    (tmp_path/'project_state.json').write_text(json.dumps(state))
    result=v.verify(tmp_path)
    assert result['result']=='FAIL' and any('Stage 1 changed' in e for e in result['errors'])
