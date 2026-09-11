"""Wait for real NVML memory release after a successful owned actor exit."""
from pathlib import Path
import subprocess
import time


def wait_for_release(read_used, threshold=4*1024**3, timeout=60, poll=.2,
                     clock=time.monotonic, sleep=time.sleep):
    started=clock();samples=[]
    while True:
        used=int(read_used());elapsed=clock()-started
        samples.append(dict(seconds=elapsed,used_bytes=used))
        if used<threshold or elapsed>=timeout:
            return dict(result='RELEASED' if used<threshold else 'TIMEOUT',seconds=elapsed,
                        threshold_bytes=threshold,timeout_seconds=timeout,samples=samples)
        sleep(min(poll,max(0,timeout-elapsed)))


def actor_directory(argv):
    if not isinstance(argv,(list,tuple)) or len(argv)<3:return None
    if Path(str(argv[1])).name!='run_stage4.py' or argv[2] not in ('actor-window','adopt-actor'):return None
    return Path(argv[argv.index('--directory')+1])


class GuardedPopen(subprocess.Popen):
    def wait(self,timeout=None):
        code=super().wait(timeout=timeout)
        directory=actor_directory(self.args)
        if code!=0 or directory is None or getattr(self,'_release_observed',False):return code
        self._release_observed=True
        import pynvml
        from medical_posttrain.rl.common import immutable,now
        pynvml.nvmlInit()
        try:
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            receipt=wait_for_release(lambda:pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            receipt.update(timestamp=now(),actor_pid=self.pid,actor_exit_code=code,
                           action=self.args[2],raw_nvml_values=True)
            immutable(directory/'gpu_release_observations'/f'{time.time_ns()}.json',receipt)
        finally:pynvml.nvmlShutdown()
        # A timeout deliberately returns the real child exit code. The original
        # frozen memory assertion still observes real NVML and fails if >=4GiB.
        return code


class SubprocessProxy:
    Popen=GuardedPopen
    def __getattr__(self,name):return getattr(subprocess,name)


def install():
    from medical_posttrain.rl import online
    online.subprocess=SubprocessProxy()
