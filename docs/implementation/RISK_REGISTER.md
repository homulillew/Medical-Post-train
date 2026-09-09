# 风险登记

2026-09-08。P0阻止受影响正式run冻结；P1必须在对应验收前关闭/有可审计处理；P2限制结论。负责人默认项目实现工程师，研究合同变化由项目owner确认。原始规划背景保留；本轮实测后的状态与范围见文末 Stage 0 更新。

| ID / 级别 | 证据或不确定性 | 触发 / 检测 | 缓解与fallback | 关闭证据 / 阶段 |
|---|---|---|---|---|
| R01 P0 | GPU可见44.99GiB，BF16微探测不是8B | real forward/backward OOM或<3GiB余量 | microbatch1、checkpointing、eager、chunk logits；必要时顺序进程；量化仅proposal | 8B memory/sleep smoke与peak记录 / S1,S4 |
| R02 P0 | torch2.11与NCCL override元数据冲突，nvcc12/runtime13 | import/链接/NCCL/sleep失败 | 独立toolchain13；先标准NCCL；如必要整体换匹配runtime栈，不隐式--no-deps | metadata+pipcheck+ABI+switch receipts / S0 |
| R03 P0 | DAPO gitlink与rolling pin不一致、新engine接口 | config.compose/import/type/单卡resource pool失败 | 固定release、minimal reviewed controller adaptation，CPU与GPU集成测试 | 两组同controller parity / S3,S4 |
| R04 P0 | 原生LoRA未加载可能回退base | adapter缺失、hash不一致、A/B无差异 | 同步barrier/强制adapter check；错误轨迹INVALID | 两轮更新后adapter/logits实证 / S2,S4,S6 |
| R05 P0 | SFT/o1与考试题可能同源 | cross-source exact/near overlap | SFT前全CMExam+CMB question exclusion，heldout优先；不删test改善分数 | dedup cluster和split hash报告 / S1 |
| R06 P0 | MedEmbed英文训练、512 encoder context | 中文UNK/低方差/否定与错解释高sem | train-only64pair诊断，CPU chunk；BGE-M3候选；不用test调weight | encoder comparison+reward decision / S2 |
| R07 P0 | thinking512可能截断最终answer | truncation>5%或closure<95% diagnostic | 两组共享1024/2048 proposal，重新算成本；不改G/5k门槛 | length/protocol freeze / S2,S4 |
| R08 P1 | all-correct hybrid std可非零 | correctness std0但adv std>0 | 同时报3种std；解释intervention删除语义排序信号 | raw group/advantage case / S4,S5 |
| R09 P0 | dynamic acceptance低/相关性高 | 32 refill batches不足8 mixed；a<0.1持续3窗口 | 保存partial与starvation，暂停诊断；不永久黑名单、不降低接受要求 | real256 refill+pilot稳定记录 / S3,S4 |
| R10 P0 | old/model/sampler logprob定义可能混淆 | 初始ratio异常、KL漂移、padding错 | 无top-p/k截断，matched temperature/mask、FP32 logprob、no dropout | numeric+gradient parity / S4 |
| R11 P0 | checkpoint看似存在但optimizer不完整 | 缺marker/hash或step/reset | 原子写、param mapping、两点保留、kill/reload测试 | interrupted-resume receipts / S1,S4 |
| R12 P1 | 崩溃后cost丢失/重复计budget | WAL计数不守恒、unknown尾部 | cost与effective lineage分离；unknown不补0，核引擎记录，必要时重做formal | ledger completeness+rollback tests / S4,S5 |
| R13 P1 | 数据字段/API变化 | CMExam train无annotation、CMB config旧示例 | source/file pinned，strict schema，均匀train抽样；不补造difficulty | schema/dedup+正式count / S1,S2,S5 |
| R14 P1 | 20k或CMB2000去重后不足 | quota无法满足 | 继续同源清洁抽取；不足BLOCKED并proposal，禁止复制/改门槛 | sample IDs/manifests / S1,S5 |
| R15 P1 | reward语义/格式套利，正答案伪推理 | top-sem wrong、重复废话、format-only高频 | 严parser、binary gate、全rawcase；reward修改须两组重开formal | controlled反例与reward bounds / S2,S4 |
| R16 P1 | 磁盘满/RAM压力/数据仅单机 | disk<reserve、RAM avail<16GiB、备份失败 | 工作额度/双写预留、限定Ray和embedding线程、hash备份 | capacity preflight+restore演练 / 全阶段 |
| R17 P1 | 时间可能从数十到数百小时 | pilot校准远超约128h工作假设 | supervisor跨会话、更新资源计划、保留预算；不可几十step结束 | calibrated预算与持久状态 / 全阶段 |
| R18 P1 | 使用test挑checkpoint/调难度 | manifest选点晚于test读取、配置漂移 | 先validation选择与hash冻结，再test；分离data access与评估进程 | comparison/selection ledger / S5,S6 |
| R19 P1 | API content/reasoning拆分或SSE chunk误计token | offline错配、TPOT分母错 | 保留raw reasoning，统一template，token/usage核对、明确chunk估计 | 100consistency+stream rows / S6 |
| R20 P2 | 单训练seed和同源考试限制外推 | 微小delta被夸大、病例cherry-pick | paired CI、cluster sensitivity、正反case、训练seed局限 | Stage5报告/claims ledger |
| R22 P2 | whole-batch单epoch可能令GSPO clip不活跃 | pilot ratio≈1/clipfrac0 | 保留sampling对照，报告限制；优化量改变另立decision | ratio记录与D-008 / S4,S5 |
| R21 P1 | CMExam README使用限制与仓库license含义不同 | 对外分发原始考试题或商业化 | 项目按研究用途；保留来源声明，Git仅结构/允许摘录；商业范围另核 | dataset provenance / S1,S5 |

所有错误、failed/invalid/negative run留manifest/日志。风险未触发不是PASS；例如没有启动服务不等于“未观察到serving mismatch所以验收通过”。

## Stage 0 实测状态更新

状态含义：OPEN=仍待证据/决策；MITIGATED=在指定边界内已有实测缓解，仍需后续gate；CLOSED=该具体风险范围已完成验证；BLOCKED=当前必须停止受影响推进。状态不替代Stage1–6验收。本轮Stage0没有留下无法运行的P0 blocker，但不批准正式训练。

| ID | 状态 | 实际证据与剩余边界 |
|---|---|---|
| R01 | MITIGATED | 真实8B BF16+r32到2048 backward通过，peak约22.69GiB；仅microbatch1/SDPA/当前optimizer设置，正式数据smoke仍须执行 |
| R02 | CLOSED（本机固定runtime） | 冻结环境、pip check、CUDA BF16、native FSDP2/NCCL、vLLM和三次sleep通过；native sampler避免CUDA12 JIT；其他GPU/toolchain不在结论范围 |
| R03 | OPEN | native import/compose/FSDP2通过；完整Ray trainer、DAPO controller和resource pool尚未运行 |
| R04 | MITIGATED | 默认V1/V2 LoRA重复logprob波动已捕获；batch-invariant/split_k1后冷/热+三次sleep一致，真实更新adapter可测；每次正式同步仍须identity断言 |
| R05 | OPEN | 未开展正式SFT配额和跨来源heldout去重 |
| R06 | OPEN | 两个encoder都把错误剂量排在同义句前，未替换；需更大train-only reward诊断 |
| R07 | OPEN | 512截断87.5%，1024仍22.656%；共同长度proposal待SFT后验证，不能直接冻结1024 |
| R08 | OPEN | 本轮没有自然on-policy group统计 |
| R09 | OPEN | natural acceptance未知，没有正式refill/pilot |
| R10 | OPEN | native GSPO numeric通过；默认LoRA数值波动进一步说明需matched-temperature跨引擎logprob parity和correction检查 |
| R11 | MITIGATED | 受控HF/PEFT adapter+Adam+scheduler+RNG+step恢复到参考结果，max参数误差0；完整dataloader与正式FSDP checkpoint恢复未做 |
| R12 | OPEN | MVP有run/attempt/checkpoint lineage，正式crash ledger/rollback计数未实现 |
| R13 | MITIGATED | 固定32条CMExam train和Qwen tokenizer实际字段/模板通过；全量数据schema仍需Stage1/2处理 |
| R14 | OPEN | 真实清洁20k/CMB2000配额未检验 |
| R15 | OPEN | semantic数字反例和格式失败已捕获，未证明reward抗套利 |
| R16 | OPEN | 本机资源充足，bulk hash存在但无外部备份，manifest不等于备份 |
| R17 | MITIGATED | 已有实测锚点与约340h条件性工作预算；正式长期吞吐/acceptance仍未知 |
| R18 | MITIGATED | 本轮只读取32条train，无test访问；正式selection/test隔离仍是后续gate |
| R19 | OPEN | 尚未执行服务SSE/100题一致性；Stage0是离线vLLM检查 |
| R20 | OPEN | 未产生正式性能提升或统计结论 |
| R21 | OPEN | 原始train32只留本机bulk，Git存manifest和少量生成案例；许可边界仍按研究用途 |
| R22 | MITIGATED | 单mini ratio=1/clip0实测，双mini第二步clip0.875；D-014为proposal，正式设置未变 |

证据入口：[Stage 0 report](../stage_reports/00_runtime_compatibility.md)、[selected runs](../../experiments/stage0/selected_runs.json)、D-009至D-016。没有观察到OOM或ABI崩溃不被写成这些失败的虚构案例。

## Stage 1 已取得证据与剩余边界（2026-09-09 CST）

本节不将进行中的formal提前记为完成；最终全程资源和生成风险以Stage1报告、完整raw metrics及收据补充。来源：[Stage1 selected runs](../../experiments/stage1/selected_runs.json)、[决策与观察](STAGE1_DECISIONS.md)。

| ID | 状态 | 本阶段直接证据与限制 |
| --- | --- | --- |
| R05 | MITIGATED（固定词法边界） | 全79,319条benchmark question-only索引，精确/近重复连通簇排除489条关联SFT成员；全21k选中编码与ID/cluster复核。语义改写漏检和近似题型误排仍可能存在，不声称零语义泄漏 |
| R11 | MITIGATED（实际SFT dataloader） | 1024例pilot在step32退出并新进程恢复，step33参数/loss差均0、next IDs/scheduler一致；formal checkpoint独立哈希检查通过。尚不覆盖未来FSDP/GSPO状态恢复 |
| R13 | MITIGATED（固定SFT原始源） | 两源全量实际schema通过，Huatuo全部单轮；原生模板与多轮fixture分别检查。数据内容医学正确性不由schema检验证明 |
| R14 | CLOSED（仅Stage1配额） | 清洁唯一候选16,646/137,370，分别选10k train+500 val，无复制或重复采样；未来CMB2000评估配额仍是独立gate |
| R18 | MITIGATED（Stage1范围） | test原始文件按字节保留，解析只访问Question/question建立预先排除；答案与难度不参与训练/选择。final-budget adapter预先固定，禁止按中途val改选；Stage5 test隔离仍须另验 |
| R16 | OPEN | 本机bulk持续保存全部有效checkpoint、原始失败和哈希，仍没有外部备份目的地；源码和小摘要push不等于模型/数据备份 |
| R06/R09/R10 | OPEN | 本阶段未运行semantic reward、自然mixed acceptance或完整GSPO actor/rollout循环；Stage0负例与后续gate仍有效 |
| R23 P2（新增） | OPEN：原始CoT事实/算术噪声 | 最长选中CoT row14099把3.9921875+4写成9.9921875，推理与末尾结论也不一致。结构清洗与去重没有审定推理正确性；保留原冻结baseline及精确算术反例。最长样本不是随机质量抽样，错误率未知；未来数据事实筛查必须另立版本/run，不能事后改写此epoch |

R23不被包装成已修复：原样本实际保留在训练数据中。后续报告只能声明完成领域SFT及实测格式/长度行为，不能凭loss下降声明医学事实质量已经提高。R07长度风险将在最终50条配对生成中进一步测量，仍需下一阶段考试train-only profiling验证迁移。

### Stage 1 全量与生成结束后的补充

正式20k/1epoch已实际完成，完整1000条验证NLL为2.055825→1.346031；R01在本次SFT配置下MITIGATED，NVML峰值37.188GiB、峰值余量7.800GiB，未发生OOM。不能将该显存数字直接用于后续同时持有actor/rollout状态的GSPO循环。

R04在最终adapter上继续MITIGATED：HF新进程重载digest一致，Base/adapter logits最大差21.8125；vLLM Base/SFT prompt-logprob最大差10.518176，SFT重复差0，真实adapter身份可核验。

R07在本次开放QA验证范围内MITIGATED：固定50条SFT在1024 cap下全部think/answer闭合且0截断，o1均有非空reasoning、Huatuo均为空think。2048未触发也未测；考试train-only profiling与正式shared generation freeze仍是后续gate。

R17的实测锚点为formal worker2.537h、更新吞吐1074.84 total tokens/s、post-SFT生成均长346.22和LoRA172.01 output tokens/s。后续工作情景约124.52h含已完成SFT，仍依赖acceptance和任务长度迁移等假设；不是已执行的项目总耗时。

R23和R20仍OPEN：实际SFT格式完整的药物回答与源参考不同，且没有专家审定；不能把格式100%或loss下降当作医学准确率提升。一个UTF-8截断边界还揭示一次性decode与原生增量decode不同；首次额外audit失败保留，按原生DecodeStream复核100条输出后通过，raw生成未改写。

## Stage 2 early observations (retained historical record)

R06/R15: controlled200-pair comparison completed; D-022 chooses BGE-M3 for bounded reference alignment and preserves0.8/0.15/0.05 correctness gating. Both encoders fail controlled negation/dose ranking; medical reasoning validity remains OPEN. Missing reference is an explicit semantic0 condition (2220/15000 pool), not an encoder error.

R04: final-adapter identity positive/negative, repeat and sleep/wake controls passed on real50×4 smoke without actor updates. Same controls required in formal. R07: smoke cap1024 has0/200 truncation,100% answer-tag closure,86.5% strict format; nested/malformed tags remain observed, longer cap is not their demonstrated fix.

R24 P2 (new): official text-only CMExam prompts can reference missing images (observed train27961 “暂无图”). Source-faithful pool is retained; report image-reference incidence and restrict interpretation to text-only exam response/label agreement. No post-hoc exclusion, no visual capability claim. R23 continues: correct final label may accompany arithmetic or reference-explanation disagreement, now retained in Stage2 qualitative worklog.

R25 (OBSERVED): four pool references exactly equal 请等待更新; one formal prompt is affected. Retain frozen results, report separately; later masking requires an explicit shared reward version decision.

Stage 2 full evidence: 1000×4 complete; final adapter controls PASS, zero truncation at1024, strict format86.925%. Source placeholders/missing images and weak semantic discrimination remain interpretation risks. No clinical validation or Stage3/4 execution is claimed.

## Stage 3 implementation and smoke evidence

R26 (OBSERVED, engineering fix under real revalidation): terminal evidence writer raised duplicate-run_id TypeError after all32 smoke groups committed. Raw generation, rewards, counters and genuine restart preserved; FAILED run remains `s3_smoke_20260909T051456_a53a8c`. A dedicated regression test covers summary merging.

R27 (OBSERVED, engineering fix under real revalidation): failed worker retained GPU allocations, so `s3_smoke_20260909T052138_e6717b` hit vLLM free-memory admission guard before any rollout. Preserve failed startup and zero generation-attempt evidence. Owned-group termination, explicit engine shutdown and prelaunch GPU-empty checks added. See D3-002/D3-003 in STAGE3_PLAN.md. No reduction of gpu_memory_utilization or scientific settings.

R28 (OPEN until formal): acc-only filtering, exact-target overflow, cyclic exposure and replay must pass raw checks for256 fresh mixed groups. Candidate starvation is bounded at128 generation batches of16; failure cannot redefine256. All rejected costs remain incurred. Future changing-policy acceptance and optimizer scheduling remain Stage4 questions.


### Stage 3 completed evidence

R26/R27 MITIGATED for the implemented workflow: third smoke passed real32x4, owned-group SIGTERM/new-process resume, raw evidence verification and clean shutdown; formal496 groups also exited cleanly. Both failed runs and costs retained. R28 MITIGATED for fixed-SFT integration:31 batches reached exactly256 accepted with1 overflow, zero invalid, full token conservation and9 full-stage verifier gates PASS. Actual formal memory peak32.606GiB; no actor resident.

Parser-only contrast remains OPEN for future training:78/257 mixed (30.3502%) are unparseable-only; accepted-only denominator78/256 (30.4688%). Formal unparseable250/1984, strict87.3992%, no truncation. The same-seed independent smoke comparison differed in22/128 raw trajectories and2/32 acc vectors; root cause remains NOT_ESTABLISHED, while per-process identity/repeat/wake controls passed. Resume relies on retained raw/commits, not regenerating completed responses.

Future Stage4 still must validate changing-policy synchronization and trainer recovery, measure actor/old-logprob/switching costs, and retain both5000-group baselines. Stage3 tests a real committed-boundary controller restart; it does not establish all in-flight, power-loss or full trainer-resume cases. R23–R25 semantic/source-quality limitations remain.

## Stage4 formal entry update (2026-09-09 15:57UTC)

R11/R12 MITIGATED for the implemented three checkpoint transaction boundaries: real SIGKILL/new-process/native LoRA+Adam+scheduler+RNG recovery and raw replay checks PASS. Effective optimizer steps6 versus physical A8/B6/C7 prove orphan work remains paid but does not inflate the valid lineage. B adopts the exact renamed checkpoint without optimizer replay. Evidence: `experiments/stage4/recovery_fault_injections.json`.

R29 OBSERVED: exact prior native state and old arrays do not guarantee bitwise GPU optimizer replay. Fault A adapter relative L2 difference2.4558e-5; second-mini max logprob difference0.38448 and max sequence-ratio difference0.008412. GPU backward/reduction nondeterminism is a hypothesis; root cause not established. Preserve both outputs, restore the trusted input state on rollback, adopt durable output when available, and disclose the limitation rather than claiming bitwise continuation.

R16 remains OPEN for external backup/power loss. Disk preflight before formal freeze reserves measured checkpoints×1250×1.1+100GiB without deleting pilots or raw evidence. R17 retains pilot-based planning estimates only; the queue has no elapsed-time success cutoff. Actual late Dynamic amplification is unknown. R09/response and clipping risks are monitored under the shared frozen bounds; starvation becomes BLOCKED, persistent clipping/length triggers diagnosis, never budget/eligibility relaxation.
