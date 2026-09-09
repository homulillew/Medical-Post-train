# Checkpoint、长期执行与恢复

目标：无论Codex/SSH是否存活，run都能依据持久证据继续原预算。此处是工程设计，本次未启动systemd服务或真正训练恢复测试。

## 脱离交互会话

未来`mpt run launch --manifest <path>`在所有preflight通过后写持久run record，再启动systemd user unit `medical-posttrain@<run_id>.service`。ExecStart为固定venv Python、固定repo revision/worktree、固定manifest的argv数组；WorkingDirectory固定，Restart=on-failure、StartLimitBurst=3（窗口15min），RestartSec=30，TimeoutStopSec足够保存，SIGTERM先请求安全边界检查点。日志同时写bulk stdout/stderr和journal，GPU全局flock防止两个formal抢同卡。tmux仅用于查看。

unit文件与enable状态跨重启保留。boot启动reconcile而不是直接盲目调用`train`：检查manifest、checkpoint、设备、数据hash、run lock；资源冲突等待，配置不符BLOCKED，确认可恢复才起原run。`Linger=yes`已实际查到，但unit创建、权限、重启流程仍须smoke验证。

state中记录unit name、host boot ID、PID、进程start time、heartbeat、run_id、checkpoint ID。单看PID可能复用；用boot ID+start time+unit核对。heartbeat每30s，超过120s触发inspect，不自动当DONE或立即杀进程。以后agent先`mpt run inspect --run-id ...`，见活任务则monitor，见死任务则reconcile，不能第二次launch覆盖它。

## 两种checkpoint用途

**Portable adapter**：PEFT adapter_model.safetensors + adapter_config.json、tokenizer/template/base refs。用于跨Stage初始化/部署。`PeftModel.from_pretrained(base,path,is_trainable=True)`是warm-start，**不包含优化器恢复**。

**Recoverable checkpoint**：adapter/model state + optimizer + scheduler + RNG + trainer/dataloader +预算/事务游标。SFT复用Trainer保存内容；RL复用verl `save_contents=[model,optimizer,extra]`。验证`save_lora_only=true`后可只在model部分存LoRA，optimizer/extra仍必须在；格式未通过roundtrip前按完整state估空间。保存module名、param shape/dtype、optimizer param-group mapping避免静默错配。

每个recoverable checkpoint附带：

```text
checkpoints/<checkpoint_id>/
  adapter/                         # 导出视图，若已生成
  actor/                           # verl model/optim/extra或SFT trainer state
  scheduler.json                   # 可审阅摘要，原始scheduler state仍保留
  rng.pt                           # Python/NumPy/torch CPU/CUDA/rank states
  stream_state.json                # 候选pool epoch/cursor/seed、group encounter序号
  counters.json                    # active lineage有效更新/数据覆盖与成本ledger offset
  transaction.json                 # committed update_id、policy_version、input/output hashes
  integrity.json                   # 每个文件sha256、size、shape inventory
  COMMITTED.json                   # 最后写入，列manifest hash与commit time
```

SFT另存shuffle排列/stream cursor、已消费sample IDs、每样本监督tokens、accumulation边界、optimizer state step。RL另存accepted training group IDs、policy update数、sampler RNG、overflow/unfinished group状态、rollout policy hash、reward version、reference embedding cache revision。

## 原子提交协议

1. 只在完整optimizer.step + scheduler.step完成后的barrier保存；所有相关worker同步，不在半个gradient accumulation边界取checkpoint。
2. 写同一filesystem的`checkpoints/.tmp-<uuid>`，同步模型/optimizer/计数/采样状态和generation ledger offset。
3. flush/fsync文件与目录，计算文件size/hash、读取关键header做结构验证；全部完成后写COMMITTED.json。
4. 原子rename到正式checkpoint_id，fsync父目录；再原子更新`latest_valid.json`（缓存指针）。扫描器以完整marker+hash为准，不依赖目录名/mtime。
5. 最后append CHECKPOINT_COMMITTED事件并发布manifest。出现崩溃允许“完整目录存在但pointer未更新”，reconcile扫描可修复；绝不接受“pointer指向半写文件”。

间隔：SFT每100updates或15min；RL每64updates（512groups）或15min；均取到达条件后的最近完整更新边界，并保存final。SIGTERM请求尽快安全保存；SIGKILL/断电依赖上个commit。checkpoint不意味着Stage完整。

## 成本账本与有效训练进度必须分开

append-only generation WAL独立于训练checkpoint。请求开始先写attempt_id、group_id、policy hash；流式/引擎侧token事件持久化（有界flush），完成时写完整token IDs/raw result/hash。filter之前提交group分类。optimizer update记录PREPARED(input group IDs,old policy)，成功后APPLIED；只有最新恢复谱系的checkpoint/后续有效commit才能用于训练budget。

崩溃例：checkpoint保存至480groups；之后又成功更新16groups、生成100groups后断电。恢复到480时，**有效训练预算恢复480**，丢失的16标orphaned updates；全部已持久100groups成本仍保留，不重置成checkpoint时的cost。重做的计算作为新attempt计成本，重复sample或恢复重放不重复计有效更新。不要用`max(log_counter,checkpoint_counter)`拼出不存在的训练进度。

每个event有单调seq、UUID、schema、wall/monotonic时间、parent transaction。raw chunks immutable；JSONL断尾只隔离残片并留hash，不删除原文件。成本查询可用SQLite WAL索引，原始事件才是源；训练effective progress沿active checkpoint lineage重算。恢复插入RECOVERY_ROLLBACK明确哪些update失效。

若硬崩溃导致最后生成tokens无法确定，记录known tokens + unknown interval/attempt，不以0或估算填实测。尽量使用引擎侧durable生成记录补齐；仍无法恢复则cost_completeness=false，相关精确token-efficiency gate失败，保留run并报告原因。完整正式证据不能由猜测修补，需要可靠补充证据或新run重做；所有浪费成本仍进入项目资源账。

## Resume入口职责

`mpt run resume --run-id R --checkpoint C`：

1. 取得run/GPU独占锁，确认没有活的旧unit；加载immutable manifest。
2. 验证checkpoint marker、hash、base/adapter configs、dataset/reward/sampler/config/code一致；环境变动需兼容验证receipt。正确性修改不能悄悄resume原科学run。
3. 初始化同一base与LoRA；load全部optimizer/scheduler/RNG，核optimizer step/param-group coverage；恢复dataloader/候选epoch与cursor。
4. 加载checkpoint有效组计数、lineage，重放成本WAL；对未提交partial batch保留成本并丢弃训练资格（或在同policy且完整时严格重放），记录处理。
5. 把恢复的adapter重新同步vLLM，清空旧KV/prefix cache与旧adapter缓存；核policy_version/hash，确认不是base-only。旧logprobs针对恢复策略重新计算，不能使用checkpoint外旧actor轨迹。
6. 写新attempt_id与恢复receipt，继续原full budget，remaining=contract−effective count。保持原run_id，started_at不改，attempt有自己的开始/结束。
7. 若全部恢复状态不足，仅有adapter：新run_id、parent_run_id、class明确warm-start；不能保留旧accepted budget声称连续formal resume。

固定代码而仅改变无科学影响microbatch可在同run的attempt config delta中记明并校验等价性；改变reward/parser/model/data/G/clip等科学配置，旧run标INVALID或独立有效旧实验，另开run，不将预算相加。

## 故障矩阵

| 故障 | 保存/检测 | 下一次行为 |
|---|---|---|
| Codex session/SSH断开 | systemd持有进程与日志 | inspect并monitor，不重启 |
| Python crash | unit exit code+stderr+heartbeat | 有限重试、最新valid checkpoint、记录attempt |
| GPU OOM | NVML峰值、allocation traceback、current phase与batch | 若仅microbatch风险，减batch/关graphs/offload并记录；清进程后恢复，不丢病例 |
| 连续OOM/NaN | 3次恢复仍失败、保留诊断 | FAILED/BLOCKED，禁止修改full最低budget |
| 机器重启 | boot ID变化、unit恢复审计 | 挂载/GPU/checkpoint完整后恢复；无证据不启动 |
| checkpoint半写/坏hash | 缺marker或hash mismatch | 隔离坏目录，回退前一valid checkpoint并重算lineage |
| 磁盘满 | reserve阈值与写失败 | 安全停止，保留至少两个valid；不得清唯一checkpoint |
| reward/泄漏/adapter错误 | verifier/在线assert | INVALID，保留全部结果，修复后新formal run |
| refill starvation | 达到32batches仍不足8groups | BLOCKED，保留partial/生成成本；不偷偷接受all-correct/wrong |

## 正式前必须通过的恢复测试

用相同初始化与固定synthetic/训练允许prompt，连续4updates与2updates→保存→结束→新进程restore→2updates对照。检查adapter/logits容差、optimizer m/v/step、scheduler、RNG、采样cursor、accepted group IDs/counters、tokens。GPU浮点差异容差在fixture预声明（参数relative error建议1e-5起审，必要时按dtype证据调整），不能仅“loss接近”就通过。随机生成跨服务重启不承诺bitwise复现，须用固定trajectory重放隔离optimizer parity，再单独测fresh-rollout恢复语义。

额外注入三种故障：checkpoint rename前kill、rename后pointer前kill、optimizer后checkpoint前kill。后者必须证明effective budget回退、cost不回退。恢复到最后checkpoint时若恰好达到5000，也仍需full verifier+report，不能直接DONE。

## Stage 0 已实现边界

`training/lora.py` 已完成真实Qwen3-8B/PEFT的step3 checkpoint与新进程step4 continuation；adapter参数digest、Adam state、scheduler、Python/NumPy/CPU/CUDA RNG均保存/恢复，对照参数和logits误差0。恢复CLI要求父run完成且checkpoint为该run已hash的可信本地artifact。文件先完整写入，再由run的checkpoint refs和artifact manifest发布；没有把一个存在但未完成的文件当成可恢复证据。

launcher使用start_new_session、stdout/stderr、PID、5秒heartbeat和有限超时，故不依赖SSH/Codex进程存活。此MVP不处理主机重启，也不实现完整formal数据cursor、systemd reconciliation、FSDP分布式状态或多点事务。Stage4前仍须补齐本文件原有正式恢复合同。Stage0受控resume不能替代后续随机中断演练。

## Stage4 smoke实测与当前边界（2026-09-09）

Vanilla和Dynamic各完成32训练groups、4windows、8optimizer steps，均在2windows后真实SIGTERM，再用新PID恢复到第4window。每次native FSDP2 actor恢复校验LoRA digest、Adam state digest/step、scheduler及显式Python/NumPy/CPU/CUDA RNG；controller/stream/生成与训练计数连续。Vanilla另已完成最终native checkpoint新进程重载与558个response-token logprob检查，Adam保持step8。

恢复证据见`experiments/stage4/{vanilla,dynamic}_smoke_verification.json`及各run的`physical_resume_receipt.json`。每window保存完整native/portable LoRA、Adam、scheduler、RNG、controller metadata和COMMITTED文件哈希，再经过vLLM新adapter ID正向sync probe才提交训练计数。launcher脱离会话、10秒heartbeat、GPU flock与重复owner防护已实现。不能把这些边界检查说成已经完成上文三种不同事务时点的真实故障注入；该项仍是formal前待验收工作。

新增`run_stage4.py recover`在确认owner退出/GPU空闲后归档未提交actor工件，保留原raw并恢复相同run；完整actor但未sync的checkpoint可复用。三项CPU恢复保护测试通过，但不替代GPU故障注入。未返回的生成/validation请求若缺raw，成本仍未知，明确拒绝静默补0重试。
