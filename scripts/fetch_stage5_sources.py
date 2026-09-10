#!/usr/bin/env python3
"""Fetch exact official CMB additions for reproducible evaluation preparation."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

REV = '6c8ece46097dae736c6805dd3b831e1a38c08971'
ROOT = Path('/data/WSH/medical-post-train-artifacts/data/stage5/raw') / REV
EXPECTED = {
    'CMB-test-choice-answer.json': ('42130b20f7660a83dd92f3c005eb6a8ef1221c98d9b27d3351204143944b532f', 2601052),
    'CMB.zip': ('c2e4288127fa5e6c03cc3659d893b0bbb19e699f74fdadce8e7fa10ac5bfdd94', 30740886),
    'CMB-Clin-qa.json': ('fc4d8d2dbfe7f647e257ceefddc9ab467576e2c4c2978fecba70f38d3c27c07e', 261255),
}


def check(path):
    expected, size = EXPECTED[path.name]
    assert path.stat().st_size == size
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    refs = []
    for name in ('CMB-test-choice-answer.json', 'CMB.zip'):
        path = ROOT / name
        url = f'https://raw.githubusercontent.com/FreedomIntelligence/CMB/{REV}/data/{name}'
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response, path.open('xb') as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        check(path)
        refs.append(dict(path=str(path), sha256=EXPECTED[name][0], bytes=EXPECTED[name][1], url=url, revision=REV))
    member = 'CMB/CMB-Clin/CMB-Clin-qa.json'
    path = ROOT / 'CMB-Clin-qa.json'
    with zipfile.ZipFile(ROOT / 'CMB.zip') as archive:
        content = archive.read(member)
        if not path.exists():
            path.write_bytes(content)
        assert path.read_bytes() == content
    check(path)
    refs.append(dict(path=str(path), sha256=EXPECTED[path.name][0], bytes=EXPECTED[path.name][1],
        archive_member=member, archive_sha256=EXPECTED['CMB.zip'][0], revision=REV))
    manifest = ROOT / 'download_manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == refs
    else:
        manifest.write_text(json.dumps(refs, indent=2) + '\n')
    print(json.dumps(dict(result='PASS', files=len(refs), revision=REV)))


if __name__ == '__main__':
    main()
