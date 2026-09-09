#!/usr/bin/env python3
"""Reconcile full historical raw rechecks with legitimate later-stage progress.

Original verifier FAIL receipts remain FAIL. This new handoff audit explicitly
replaces only their obsolete NOT_STARTED assertions, and repeats their remaining
integrity assertions, after requiring every scientific raw gate to have passed.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,immutable,validate,inherited,now
from medical_posttrain.evidence import sha256


def main():
    state = read(ROOT/'project_state.json')
    assert state['stage0']['status']=='VERIFIED'
    assert all(state['stages'][str(s)]['status']=='DONE' for s in (1,2,3))
    assert all(state['stages'][str(s)]['status']=='NOT_STARTED' for s in (5,6))
    expected_exceptions = {1:'research_contract_and_stage_isolation',2:'prior_stages_and_unchanged_contract_isolation'}
    fresh = []
    for s in (1,2,3):
        path = INDEX/f'prerequisite-stage{s}.json'
        receipt = read(path)
        failed = {k for k,v in receipt['gates'].items() if not v}
        assert failed == ({expected_exceptions[s]} if s in expected_exceptions else set())
        fresh.append(record(path))
        old = state['stages'][str(s)]['verification_receipt']
        assert sha256(old['path'])==old['sha256'] and read(old['path'])['result']=='PASS'
    assert sha256(ROOT/'contracts/stage_budgets.json')==state['contract_sha256']
    for p,h in read(ROOT/'contracts/stage_budgets.json')['source_documents'].items():
        assert sha256(ROOT/p)==h
    assert read(ROOT/state['stage0']['verification_receipt'])['result']=='PASS'
    base = read(ROOT/'experiments/stage0/s0_snapshot_20260908T132515_ec08e3/attempt_001/snapshot_manifest.json')
    assert base['revision']=='b968826d9c46dd6066d109eabc6255188de91218'
    for f in base['files']:
        assert sha256(f['path'])==f['sha256']
    initial = validate(inherited())
    for f in initial['files']:
        assert sha256(f['path'])==f['sha256']
    result = dict(result='PASS',scope='Stage4 predecessor handoff; does not rewrite historical FAIL receipts',
        timestamp=now(),fresh_raw_checks=fresh,stage_statuses={str(s):state['stages'][str(s)]['status'] for s in range(1,7)},
        superseded_assertions=expected_exceptions,
        reason='Stage1/2 verifiers require all later stages NOT_STARTED, incompatible with verified Stage2/3 completion',
        scientific_raw_gates_all_pass=True,contract_and_base_and_adapter_rehashed=True)
    immutable(INDEX/'prerequisite-handoff.json',result)
    print(result)


if __name__=='__main__':
    main()
