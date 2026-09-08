"""Recover one pinned official shard using bounded validated HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path('/data/WSH/medical-post-train-artifacts/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218')
NAME='model-00002-of-00005.safetensors'
SIZE=3993160032
SHA='5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282'
URL='https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/'+NAME

def main():
    parts=ROOT.parent/'shard2-recovery-parts';parts.mkdir(exist_ok=True)
    block=16*1024*1024
    def fetch(start):
        end=min(start+block,SIZE)-1
        p=parts/f'{start:012d}.part'
        if p.exists() and p.stat().st_size==end-start+1:return
        for attempt in range(5):
            try:
                request=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=90) as r:
                    assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{SIZE}'
                    data=r.read(end-start+2)
                assert len(data)==end-start+1
                tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.rename(p)
                print(json.dumps(dict(start=start,end=end,status='downloaded',attempt=attempt+1)),flush=True)
                return
            except Exception as exc:
                print(json.dumps(dict(start=start,attempt=attempt+1,error=str(exc))),flush=True)
                if attempt==4:raise
    starts=list(range(0,SIZE,block))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(fetch,s) for s in starts]):f.result()
    tmp=ROOT/(NAME+'.range-recovery')
    h=hashlib.sha256()
    with tmp.open('xb') as out:
        for s in starts:
            data=(parts/f'{s:012d}.part').read_bytes();h.update(data);out.write(data)
    assert h.hexdigest()==SHA
    assert not (ROOT/NAME).exists()
    tmp.rename(ROOT/NAME)
    print(json.dumps(dict(status='VERIFIED',sha256=h.hexdigest(),bytes=SIZE)),flush=True)

if __name__=='__main__':main()
