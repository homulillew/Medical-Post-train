# Stage 4 GPU teardown incident and operational recovery

## Observation

Vanilla run `s4_formal_vanilla_20260909T155928_24aec5` failed at
2026-09-10 17:56:14 Asia/Shanghai. The frozen controller asserted
`Actor failed to release GPU` immediately after actor subprocess exit.
The actor for window `0402` exited with code 0 and retained its checkpoint,
Adam state, scheduler and RNG after optimizer steps 805 and 806. The last synced
and committed state was 3,216 groups / 402 windows / 804 optimizer steps.
The extra eight groups are not counted as committed until recovery sync passes.

At inspection on September 11 around 10:18, no worker or queue process remained,
GPU memory was 15 MiB, and the queue recorded NEEDS_DIAGNOSIS. The project-level
FULL_RUNNING field was stale. Training had therefore been idle for roughly
16 hours 22 minutes; yesterday's running-rate ETA did not describe actual overnight
progress. The failure and stale status are retained as evidence.

Delayed CUDA/driver cleanup after process exit is consistent with these observations.
The failed assertion did not retain the exact offending NVML memory value or its
release timeline, so a specific driver defect or exact delay is not established.

## Operational change

`stage4_runtime_v2.py` uses the existing boundary restore and an explicit
`stage4_gpu_release_guard.py` wrapper around successful actor/adoption subprocess
waits. It reads real NVML memory until usage is below the original 4 GiB threshold,
or 60 seconds elapse, polling every 0.2 seconds. The ordinary already-released
case adds no sleep. Readings and elapsed time are retained per window.

The guard does not modify or fake memory values. On timeout, the original frozen
assertion still sees actual NVML usage and can fail. No external GPU process is
killed. Training loss, reward, sampling, learning rate, group budget, checkpoint
format and model initialization remain byte-identical to the frozen formal pair.

`stage4_queue_runtime.py` runs the unchanged formal queue, substituting the same
operational launcher for both Vanilla and Dynamic. Full verification remains
mandatory before queue advancement. Queue exceptions now also mark the project
status FAILED, avoiding a stale FULL_RUNNING label on a known queue failure.
This is an engineering change, not a project-contract or scientific change.

## Recovery requirements and evidence

Retain the original failure and current window's raw rollouts, selection, update,
actor result, checkpoint marker and controller before any recovery mutation.
Use the fault-tested complete-checkpoint adoption path: verify the commit chain,
full bytes of the last load boundary and the pending checkpoint, then reload
pending native optimizer/scheduler/RNG and sync the adapter without optimizer
replay. Historical checkpoint payloads remain retained for the mandatory final
full verification; they are not all reread during operational restore.

Seventeen unit checks passed before recovery: delayed/timeout/immediate release,
actor-only scope, queue routing, boundary equivalence and corruption/gap checks.
Successful launch alone is insufficient. Verify adoption adds zero optimizer
steps, the resumed commit advances exactly eight groups/two already-completed
steps, and its original response/selection/update/checkpoint hashes stay unchanged.
Then observe a fresh subsequent window with the actual GPU-release guard active.

API evaluation remains stopped under the user's budget instruction. This recovery
does not issue API calls or run project-model final-test evaluation.

## Verified recovery and progress — September 11, 10:31

The original run resumed as `attempt_003`, worker PID 2811475, with persistent
queue PID 2811546. Complete native checkpoint adoption passed optimizer, scheduler
and RNG checks and committed window0402 at 3,224 groups / 806 steps with **zero
optimizer steps replayed**. All 16 pinned pending-rollout/selection/update/control
references remained unchanged. The next fresh window0403 then committed at
**3,232 / 5,000 groups (64.64%)**, 404 windows / 808 optimizer steps.

Both real teardown observations returned RELEASED with raw NVML readings below
the original threshold. The timeout/delayed-release branches have unit-test
coverage; the old timing fault was not reproduced during these two observations.
This is evidence of successful recovery and continued training, not a claim that
every possible future GPU failure has been eliminated. All 121 original source
hashes, including the 49 frozen execution files, remain unchanged.

Boundary restores checked the full commit chain plus the latest checkpoint bytes
in approximately 3.4–3.6 seconds. The pending checkpoint was separately verified
and reloaded; historical checkpoint payload hashing remains a final-acceptance
requirement. The initial audit supervisor launch accidentally resolved the venv
Python symlink to the base interpreter; it was stopped before any worker/attempt
creation and relaunched through the correct venv path. Both launch records and
the correction are retained.

Latest completed monitor512 was at 3,072 groups: **64.0625%**, compared with
**55.078125%** SFT initialization (+8.984375 percentage points). Recent points were
65.234375% at2,496 and63.28125% at2,560. The recent63–65% range does not establish
continued improvement or a final plateau. Dynamic remains0/5,000, so the principal
sampling-intervention comparison is still unavailable.

Before the outage, the last20 committed windows averaged63.125% training-sample
correctness, with5/640 unparseable responses and1/640 truncated response. These
training metrics are not held-out test accuracy. The same20-window wall interval
delivered205.85 groups/hour including a scheduled monitor, versus roughly265/hour
in uninterrupted training. Remaining Vanilla training and routine validation are
roughly7–9 hours if those speeds persist; final reload/full verification and the
entire Dynamic run are additional. No ETA can recover the overnight downtime.

Evidence:

- [Incident and preserved sources](../../experiments/stage4/vanilla_formal_incident_002.json)
- [Checkpoint adoption verification](../../experiments/stage4/vanilla_formal_resume_verified_002.json)
- [Fresh guarded window verification](../../experiments/stage4/vanilla_formal_fresh_guard_verified_002.json)
- [Current recovery status](../../experiments/stage4/gpu_incident_recovery_status.json)
- [Test receipt](../../experiments/stage4/gpu_release_guard_tests_001.xml)
