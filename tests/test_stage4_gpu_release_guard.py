from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from stage4_gpu_release_guard import wait_for_release,actor_directory


class Clock:
    t=0
    def now(self):return self.t
    def sleep(self,n):self.t+=n


def test_delayed_release_waits_for_real_low_memory():
    clock=Clock();values=iter([8,7,4,3]);r=wait_for_release(lambda:next(values),threshold=4,timeout=2,poll=.2,clock=clock.now,sleep=clock.sleep)
    assert r['result']=='RELEASED' and r['seconds']==pytest.approx(.6)
    assert [s['used_bytes'] for s in r['samples']]==[8,7,4,3]


def test_persistent_residue_is_not_hidden():
    clock=Clock();r=wait_for_release(lambda:8,threshold=4,timeout=1,poll=.2,clock=clock.now,sleep=clock.sleep)
    assert r['result']=='TIMEOUT' and r['seconds']==pytest.approx(1)
    assert all(s['used_bytes']==8 for s in r['samples'])


def test_already_released_does_not_delay():
    clock=Clock();r=wait_for_release(lambda:3,threshold=4,clock=clock.now,sleep=clock.sleep)
    assert r['seconds']==0 and len(r['samples'])==1


def test_only_actor_and_adoption_receive_guard():
    assert actor_directory(['python','/repo/run_stage4.py','actor-window','--directory','/run/window'])==Path('/run/window')
    assert actor_directory(['python','/repo/run_stage4.py','adopt-actor','--directory','/run/window'])==Path('/run/window')
    assert actor_directory(['python','/repo/run_stage4.py','worker']) is None
    assert actor_directory(['python','/repo/other.py','actor-window']) is None


def test_queue_routes_only_frozen_worker_launch(monkeypatch):
    import stage4_queue_runtime as queue
    import stage4_runtime_v2 as runtime
    seen=[];monkeypatch.setattr(runtime,'launch',lambda out:seen.append(out) or 123)
    cmd=['python',str(queue.ROOT/'scripts/run_stage4.py'),'launch','--run','/run/vanilla']
    assert queue.QueueSubprocessProxy().run(cmd,check=True).returncode==0
    assert seen==[Path('/run/vanilla')]
