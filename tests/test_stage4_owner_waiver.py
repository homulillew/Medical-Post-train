"""CPU checks for the narrowly scoped owner waiver and unchanged scientific gates."""
import ast
import copy
import json
from pathlib import Path

import pytest
import verify_stage4 as verifier

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    docs = {
        'cases': {'manual_reviews': []},
        'waiver': {
            'decision': 'OWNER_WAIVED',
            'affected_gate': 'stage4.manual_response_review_count_ge_2',
            'waiver_scope': 'STAGE4_DOCUMENTATION_MANUAL_REVIEW_ONLY',
            'human_reviews_completed': 0,
            'manual_case_packet_retained': True,
            'scientific_evidence_unchanged': True,
            'clinical_validation': False,
            'is_human_review': False,
            'manual_cases': {'path': 'cases'},
            'timestamp': 'synthetic', 'current_head': 'a' * 40,
            'scientific_refs': [{'path': 'science'}],
        },
    }
    instruction = tmp_path / 'owner.txt'
    instruction.write_text('Stage4 不再要求人工 response review 作为 blocking gate')
    for key in ('prior_verifier', 'manual_case_packet', 'start_audit'):
        docs['waiver'][key] = {'path': key}
    docs['waiver']['owner_instruction'] = {'path': str(instruction)}
    monkeypatch.setattr(verifier, 'read', lambda p: docs[str(p)])
    monkeypatch.setattr(verifier, 'record', lambda p: {'path': str(p)})
    return docs, {'cases': {'path': 'cases'}, 'manual_review_waiver': {'path': 'waiver'},
                  'human_review_status': 'OWNER_WAIVED', 'clinical_validation': False}


def test_valid_waiver_does_not_create_human_reviews(fixture):
    docs, manifest = fixture
    before = copy.deepcopy(docs)
    verifier.verify_manual_review(manifest)
    assert docs == before and docs['cases']['manual_reviews'] == []


@pytest.mark.parametrize('key,value', [
    ('decision', 'AGENT_WAIVED'), ('waiver_scope', 'ALL_GATES'),
    ('human_reviews_completed', 2), ('scientific_evidence_unchanged', False),
    ('clinical_validation', True), ('is_human_review', True),
    ('affected_gate', 'formal_budget'),
])
def test_invalid_waiver_refused(fixture, key, value):
    docs, manifest = fixture
    docs['waiver'][key] = value
    with pytest.raises(AssertionError):
        verifier.verify_manual_review(manifest)


def test_scientific_hash_mismatch_refused(fixture, monkeypatch):
    _, manifest = fixture
    monkeypatch.setattr(verifier, 'record', lambda p: {'path': str(p), **({'tampered': True} if p == 'science' else {})})
    with pytest.raises(AssertionError):
        verifier.verify_manual_review(manifest)


def test_original_human_review_route_preserved(fixture):
    docs, manifest = fixture
    docs['cases']['manual_reviews'] = [{'read_in_full': True, 'observation': 'Synthetic test only', 'sources': []}] * 2
    manifest.pop('manual_review_waiver')
    verifier.verify_manual_review(manifest)
    docs['cases']['manual_reviews'][0]['read_in_full'] = False
    with pytest.raises(AssertionError):
        verifier.verify_manual_review(manifest)


def test_only_manual_review_logic_changed():
    old = (ROOT / 'experiments/stage4/closure_source_snapshots/verify_stage4_before_owner_waiver.py').read_text()
    current = (ROOT / 'scripts/verify_stage4.py').read_text()
    start = current.index('def verify_manual_review(')
    end = current.index('def verify():', start)
    current = current[:start] + current[end:]
    old_block = """        reviewed=read(manifest['cases']['path'])
        assert len(reviewed['manual_reviews'])>=2
        for case in reviewed['manual_reviews']:
            assert case['read_in_full'] and case['observation']
            for ref in case['sources']:
                assert record(ref['path'])==ref
"""
    assert old_block in old
    old = old.replace(old_block, '        verify_manual_review(manifest)\n')
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(current))
