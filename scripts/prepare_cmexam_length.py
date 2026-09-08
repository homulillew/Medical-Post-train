"""Read precisely 32 train CSV records from a fixed upstream revision, then close."""
import csv
import io
import json
from pathlib import Path
import urllib.request
from medical_posttrain.evidence import sha256,write_json

URL='https://raw.githubusercontent.com/williamliujl/CMExam/fadb22c89beb1b7115dc36460ba792eb96b7b972/data/train.csv'
DEST=Path('/data/WSH/medical-post-train-artifacts/data/cmexam-train32-fadb22c8.json')

def main():
    if DEST.exists():raise FileExistsError(DEST)
    with urllib.request.urlopen(URL,timeout=90) as response:
        reader=csv.DictReader(io.TextIOWrapper(response,encoding='utf-8-sig'))
        rows=[next(reader) for _ in range(32)]
    write_json(DEST,dict(source=URL,split='train',selection='first 32 CSV records; no label filtering',rows=rows))
    write_json('experiments/stage0/bootstrap/cmexam-train32-manifest.json',dict(path=str(DEST),sha256=sha256(DEST),source=URL,split='train',count=32,selection='first 32 CSV records; no label filtering'))

if __name__=='__main__':main()
