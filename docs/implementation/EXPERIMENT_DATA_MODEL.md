# Run、artifact、state 与研究记忆

## 身份与目录

run_id=`s<stage>_<purpose>_<YYYYMMDDTHHMMSSZ>_<configsha8>_<uuid8>`；本次stage0环境probe用同等唯一性格式。`run_class`严格使用协议的SMOKE/PILOT/FORMAL/DIAGNOSTIC/EXPLORATORY/EVALUATION/SERVING_BENCHMARK。另设mode=smoke/pilot/full与purpose=environment_probe等，避免增加含糊的“full class”。Stage5 full为EVALUATION，Stage6 full为SERVING_BENCHMARK，stage0为DIAGNOSTIC。

同一次有效中断恢复：同run_id、新attempt_id；改科学配置：新run_id与parent_run_id，不能复用目录。`run prepare`原子创建，不允许覆盖已存在ID。

```text
<bulk>/runs/<run_id>/
  manifest.json                  # immutable身份、目标、预算、比较条件
  config.yaml / config.resolved.json
  command.json / command.txt     # argv不经shell二次解释
  environment.json / code.json   # commit、dirty patch hash、依赖lock hash
  status.json / heartbeat.json
  events/000000.jsonl            # append-only、有序、含失败attempt
  metrics.jsonl                  # 聚合视图：不替代events
  generations/*.parquet          # prompt/group/trajectory；原始token/text
  updates/*.json                # 输入groups/oldpolicy/optimizertransaction
  validation/ / evaluation/
  checkpoints/ / exports/
  cases/index.jsonl
  observations.md
  attempts/<attempt_id>/{stdout.log,stderr.log,exit.json}
  artifacts_manifest.json
  summary.json / run_report.md
  verification/{checks.json,receipt.json}
```

Git：`experiments/index/<run_id>.json`、compact curves、case excerpts/refs、observations/decisions/stage reports/interview/claims、artifact manifests。bulk：完整raw输出、token IDs、logprobs、optimizer、weights、完整stderr。每个artifact有ID/kind/run_id/URI/size/SHA256/created_at/source_commit/storage_tier/durability_status；URI失效就验收失败，不能只验证manifest存在。

## 数据实体（未来Parquet/JSON schema）

| 实体 / 主键 | 核心字段 |
|---|---|
| DatasetManifest / dataset_id | source ID/revision、file hash、license note、selector commit/seed、split count、sample ID hash、dedup settings/cluster refs |
| Prompt / prompt_id | source logical row、question/options、split、cluster、annotations；ground_truth/reference为受控reward/eval字段 |
| GenerationAttempt / attempt_id | run/policy/group/trajectory IDs、request args、seed、started/end、error/finish_reason、known/unknown generated tokens |
| Trajectory / trajectory_id | raw text/token IDs、prompt token hash、mask、parsed answer/method/ambiguity、acc/sem/format/total、encoder chunk stats、timing |
| Group / group_id | prompt encounter/policy version、G、trajectory IDs、acc vector、class、hybrid/adv std、eligibility、discard reason |
| Update / update_id | input group IDs、policy before/after、old_logprob hash、optimizer step、loss metrics、transaction status、checkpoint lineage |
| Checkpoint / checkpoint_id | run/config/code/data hashes、policy version、files+optimizer/schema、effective counters、ledger offset、COMMITTED marker |
| Prediction / (eval_run,checkpoint,prompt_id,replicate) | generation settings/seed、raw output、answer/correct、length、finish reason、error |
| Case / case_id | run/stage/category/subtype、related prompt/groups/checkpoints、raw artifact offsets、observation/hypothesis/alternatives/followup |
| VerificationReceipt / receipt_id | verifier code SHA、contract hash、inputs/artifact hashes、per-clause pass/fail、recomputed counters、timestamp、report refs |

浮点NaN/Inf不可序列化为合法指标；未知用null+missing_reason，不填0。schema_version不兼容时显式迁移保留旧文件。raw text/病题仅按来源允许范围提交compact摘录；本次不复制测试题进Git。

## 计数定义与守恒

- generated_attempts包括失败/重试；completed_trajectories包括length-finished响应，排除transport失败。
- generated_groups指已发起完整G请求的encounter；valid_generated_groups指全部G有效完成、同policy且可评分的组。
- `valid_generated_groups = all_correct + all_wrong + mixed`；另有invalid/inflight group。
- Dynamic `eligible_mixed = accepted_training + eligible_overflow + eligible_unconsumed`（按最终disposition去重）；Vanilla eligibility是所有valid组。
- accepted_training_groups只计算进入active lineage成功优化的组；不能将refill选中但未训练的组计入Stage4预算。Stage3无optimizer，用单独`accepted_integration_groups`。
- trajectories_by_update恰为G×accepted_training_groups；groups_per_update固定8。accepted group count与policy_updates同样按有效lineage重算，不能只读summary数字。
- prompt_tokens：既存logical prompt tokens一次/encounter，又存expanded input tokens一次/trajectory；KV cache命中不改变逻辑token数，cached tokens单列。
- output tokens：包括thinking、answer、EOS计数口径明确（以engine token_ids为准），所有rejected/overflow/retry已知token也计费；validation/serving另分purpose。
- wall clock分run跨度、各attempt wall、GPU reserved、generation/oldlogprob/train/reward/checkpoint时间；resume停机间隔不混入kernel吞吐。

每update记录reward各分量、entropy/proxy定义、clip fraction、grad norm、response长度分布、group比例（**过滤前**）、sampling amplification、累计cost/effective budgets、peak memory、validation checkpoints。sampling amplification同时报告valid-generated/accepted与attempt-generated/accepted，禁止换分母美化。

## State与验收

`project_state.json`是可审阅索引，当前stage1..6全NOT_STARTED。状态依协议；run另有CREATED/RUNNING/INTERRUPTED/COMPLETE/FAILED/BLOCKED/INVALID，`validity`为UNKNOWN/VALID/INVALID。run COMPLETE只表示运行器完成其声明任务，不等于stage DONE。

JSON schema可以约束状态名、数字、receipt字段与最低budget，但**schema不能证明文件真的存在或内容真实**。未来`verify_stage.py --stage N --project-state ... --output receipt.json`负责：

1. 对照`contracts/stage_budgets.json`与阶段Markdown hash，拒绝自定义降低阈值；若合同改动要求decision ID与owner授权记录。
2. 检查implementation/tests/smoke/pilot或合法N/A、前序stage的verified artifact、run_class与immutable formal manifest先于执行时间。
3. 扫raw IDs/events/checkpoint重新计数，排除重复、orphan、invalid和pilot；核对生成成本完整性与所有文件hash。
4. 执行/核验reload与resume receipts、matched baseline controls、test integrity、每个Stage特有证据。
5. 输出每条criterion PASS/FAIL和原因，不能“若无数据就skip并pass”；无指标改善不构成失败。
6. 状态写入器重跑verification并核receipt input hashes，FULL_PASS/VERIFIED/DONE逐级推进；DONE再校report/interview/cases/claims refs齐全。

GPT/agent不能凭自然语言或改JSON把stage标DONE。CI和后续stage入口**每次重新运行verifier**，不信任状态文件自称DONE；receipt与contract修改受code-review流程约束。拥有文件写权限的人仍能修改程序，因此这是可审计工程gate，不声称密码学上防御恶意仓库管理员。

| Stage | 必检正式条件 | 必检留证 |
|---|---|---|
| 1 | train=20000、目标完整epoch、coverage>=99%、无无效损失 | data/dedup/token stats、loss、final reload、val sanity、报告/2–5 cases |
| 2 | pool=15000、>=1000 unique prompts每题G4、>=4000完整responses | parser单测/人工审计、四分量/类别/cost、raw profiling |
| 3 | >=256 real accepted mixed、on-policy refill、G4完整 | 16 patterns/total-reward-independent测试、守恒计数、starvation记录 |
| 4 | 两份独立FORMAL各>=5000训练groups/20000trajectory、Dynamic全mixed、同SFT/reward/config | 两组恢复证据、raw成本与effective进度、validation、reload、confounders |
| 5 | 三模型×6811 CMExam与×2000 clean CMB，覆盖同ID | difficulty/paired CI、三类曲线、token-vs-update、正反cases/open-ended |
| 6 | validation选adapter、>=100consistency、并发1/4/8/16 | >=100请求合同下限，计划100/条件；TTFT/TPOT/E2E/throughput/VRAM，API与runbook |

Stage1 pilot可N/A、Stage2/3/5/6可选pilot也可N/A；state transition中不造PILOT_PASS，记录gate waiver reason/contract clause后进入FULL_RUNNING。Stage4本计划采用pilot并强制真实resume；没有自动optional waiver。

本次`verify_planning.py`仅校这次文档/JSON、原合同hash、全部Stage未开始及schema负例；**不叫Stage verifier，也不发VERIFIED/DONE receipt**。

## Case capture与自动挖掘

每次生成先保留完整raw row，再异步case mining；不能只存top-k代表样本而丢总体。case ID不可覆写，修订解释追加version/parent。top-k用于展示，增加seeded reservoir覆盖普通样本以减少cherry-picking。

| 触发/配对 | 自动记录 |
|---|---|
| parser ambiguous、格式塌缩、极长/截断、reward不在范围 | BAD_CASE、REWARD_CASE；原text/span、components、version |
| sem高但acc=0、sem低但acc=1、重复解释/否定反例 | REWARD_CASE；gated/ungated分析值分开，后者不进入训练 |
| all-correct/all-wrong/mixed、1/4或3/4正确 | SAMPLING_CASE、BOUNDARY_CASE；完整G4、接受/拒绝原因 |
| 同prompt跨policy类别变化 | frontier case；同generation policy证据，否则标confounder |
| SFT wrong→Dynamic correct / SFT correct→Dynamic wrong | GOOD_CASE / REGRESSION_CASE，paired prompt ID与checkpoint hash |
| Vanilla wrong→Dynamic correct / Vanilla correct→Dynamic wrong | 对称挖掘/展示；按difficulty/category/length分层 |
| entropy突降、gradnorm/clip异常、NaN/OOM/checkpoint损坏 | SYSTEM_CASE；之前/之后update窗口、日志offset、尝试修复与结果 |
| offline-serving mismatch、API格式错、latency outlier | SYSTEM_CASE；workload/request/tokenized prompt/版本/stream事件 |

initial mining每类最多20条top/rank +20 reservoir，阶段report选2–5条，并列总体发生率；没观察到某类型就写zero/not observed，不编病例。reward hacking只能先标suspected，定论需反例或controlled diagnostic。difficulty/model frontier是观察标签，不反向进入test-tuned sampling。

## Research Memory作为流水线输出

- Observation Log：每run创建observations.md；阶段汇总`experiments/observations/O-*.md`沿用模板，必须有observation/evidence/hypothesis/alternative/follow-up/conclusion，未测字段明确UNRESOLVED。在线阈值事件可自动起草，agent补解释需引用event/case。
- Decision Records：`experiments/decisions/D-*.md`保存日期、status、contract_change、备选、证据、比较与资源影响、follow-up；LR/length/rank/API/版本/OOM路径变化会阻止formal config freeze直到有关decision存在。
- Stage Report：verifier已达预算后生成待填写report skeleton，回填全部真实run（含失败）与acceptance table、2–5 cases、局限、30秒/2分钟/深挖；不在本次提前编写阶段成功故事。
- Interview/Resume Evidence：claims ledger逐条candidate claim→run/artifact/metric path→conditions→verified status。缺Stage证据时“完成20k SFT/GSPO/部署”都不安全。本次最多声明完成规划与环境审计，不声明模型效果。
- 报告、case、decision之间双向链接，CI查路径与artifact hash；证据缺失时保留MISSING，不回填推测数字。

## Stage 0 review addition: unique prompt exposure

Stage 4 records `generated_unique_prompts`, `accepted_unique_prompts`, `prompt_repeat_histogram`, and `max_prompt_exposure` under `schemas/prompt_exposure.schema.json`. Prompt identity is the immutable train record ID, independent of rollout group UUID; a repeated cyclic draw increments exposure once per prompt group, not four times for G=4. Histograms and maxima have separate generated and accepted views. Rejected/overflow groups count toward generated cost; only groups consumed by an optimizer update count toward accepted effective exposure.

The future Stage 4 verifier must recompute these from raw prompt/group/update lineage: sum(histogram frequencies)=unique prompts; sum(exposure*frequency)=generated or accepted groups; maximum occupied bin=max exposure; accepted unique <= generated unique. Crash attempts retain physical generated cost, while effective accepted lineage excludes rolled-back updates. Report per-window and cumulative distributions for both conditions. No synthetic Stage 0 exposure is backfilled as a formal measurement.
