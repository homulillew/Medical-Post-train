# 仓库架构与模块边界

本文件定义未来实现接口；除本次环境探测、planning 校验、JSON schema 外，所列代码尚未实现。避免空 trainer 或成功返回的 placeholder。单包名 `medical_posttrain`，src layout，训练入口统一由 run controller 包装。

```text
configs/
  project.yaml                 # 路径、默认 seed、显存预留
  data/{sft,cmexam,cmb}.yaml    # pinned revisions、split、dedup
  model/qwen3_8b_lora.yaml      # 唯一 backbone/adapter 定义
  stages/{s1,s2,s3,s4,s5,s6}.yaml
  modes/{smoke,pilot,full}.yaml
  comparisons/gspo_primary.yaml
contracts/stage_budgets.json    # 当前研究预算；不得被 modes 覆盖
schemas/                       # 状态、run、证据字段
project_state.json             # Git 索引；当前六阶段 NOT_STARTED
src/medical_posttrain/
  cli.py                       # argparse 子命令，组合配置、启动 controller
  data/{sources,normalize,dedup,splits,tokenize,manifests}.py
  training/{sft,collator,verl_bridge,gspo_controller}.py
  reward/{parser,correctness,semantic,hybrid,verl_score}.py
  rollout/{protocol,vllm_backend,weight_sync,records}.py
  sampling/{groups,refill,stream}.py
  evaluation/{predict,selection,paired,buckets,plots,overlap}.py
  serving/{export,launch,client,benchmark,consistency}.py
  runtime/{run,events,counters,checkpoint,supervisor,resources}.py
  evidence/{manifest,cases,observations,claims,reports}.py
  verification/{common,s1,s2,s3,s4,s5,s6,state}.py
scripts/
  probe_environment.py         # 已实现：无模型的环境检查
  verify_planning.py           # 本次规划与初始状态检查，不是 Stage verifier
  verify_stage.py              # 待实现：阶段证据验收 CLI
  jobs/medical-posttrain@.service
  jobs/reconcile.service
  bootstrap.sh                 # 待候选环境验证后实现，不自动 full run
  resume_run.py
  monitor_run.py
tests/{unit,integration,acceptance}/
experiments/{index,comparisons,decisions,observations,cases,reports,claims}/
artifacts_index/                # 小 manifest：URI、hash、size、持久性
```

## 关键接口

| 模块 | 输入 → 输出 / 职责 | 复用边界 |
|---|---|---|
| sources | 固定 ID/revision/file → 本地 immutable raw manifest | HF snapshot/JSON/CSV reader；不运行远程 dataset code |
| normalize | 原字段 → SFTExample / ExamPrompt | 仅自有字段映射与质量校验 |
| dedup / splits | question fingerprint、来源 → cluster、exclude reasons、split IDs | 规范化 + char ngram MinHash 候选 + 精确复核 |
| tokenize / collator | messages + tokenizer hash → token IDs、assistant mask、统计 | Qwen tokenizer；prompt labels=-100，PAD 与 EOS 分开处理 |
| sft | train/val manifest + config → adapter + complete trainer state | Transformers Trainer + PEFT；自有 sample coverage callback |
| parser | 原始生成 token/text + options → ParseResult | 自有严格/回退 parser，无 LLM judge |
| semantic | reasoning/reference → cosine、normalized score、截断信息 | SentenceTransformer；CPU 默认，参考向量缓存 |
| verl_score | prompt metadata + completion → score/acc/sem/format | 固定 verl reward-loop adapter |
| vllm_backend | policy_version + prompt groups → trajectories | 上游 async rollout manager；同步 policy barrier |
| groups / refill | 完整 G=4 trajectories → 分类与 update batch | DAPO 的 correctness-std 语义，强化 bookkeeping |
| verl_bridge | 项目 config → resolved Hydra + worker setup | 固定 release 的 engine workers/FSDP2，禁用无关 critic/reference |
| gspo_controller | BudgetContract + group stream → 更新/检查点 | 复用 GSPO loss、GRPO advantage、actor optimizer；自有预算/事务/审计 |
| paired / plots | 固定 predictions + ledger → 表、CI、图源 | scipy/bootstrap、matplotlib；不重跑生成来填缺值 |
| benchmark | workload + streaming client events → request rows、分位数 | vLLM API；自有请求级 timestamps/token accounting |
| verification | contract + manifests + raw evidence → receipt | 重新计算计数与 hash，不信 stdout 的“success” |

`RolloutBackend.generate(groups, policy_version)` 保证每个 group_id 对应一个 prompt encounter 与 G 个唯一 trajectory_id。`RefillController.collect(target_groups, policy_version)` 仅在同一 policy_version 内 refill。`CheckpointManager.commit(update_state, ledger_offset)` 只能在 optimizer 成功完成边界调用。`StageVerifier.verify(stage, evidence_index)` 是只读验证；独立 state writer 消费可复算 receipt 后更新索引。

## 控制与数据流

```text
固定 raw 数据 → 标准化/去重/隔离 → manifests → tokenized shards
                                              ↓
                 run controller → resolved config / budget / code / environment
                      ↓
SFT trainer → SFT adapter + optimizer state → Stage 1 verifier → artifact ID
                      ↓
rollout(current policy) → raw trajectories → parser + reward components
                      ↓                         ↓
               durable cost ledger      correctness group filter/refill
                                                ↓
                                actor old-logprob → group advantage → GSPO
                                                ↓
                                optimizer transaction → checkpoint → policy sync
                                                ↓
                                 validation-only selection → final eval → serving
```

每一条 raw trajectory 在过滤前持久记录，包括无效/全错/全对/overflow。metrics 由 event ledger 聚合，case miner 从同一行索引抓取；报告和 plots 只读已固化输入，不触发隐式训练。重要模型交接使用 artifact_id + SHA256，不使用可变 `latest/`。

## 配置与路径

合并顺序：project → model/data → stage → mode → comparison → 允许的 CLI engineering overrides。最终先 resolve、schema validate、写 canonical JSON/hash，再起 GPU。full 的 20k、15k、G=4、5k 等字段来自 contract，命令行降低预算直接报错。smoke/pilot 用新 run_id 和不同 class，不可改名升级为正式实验。每个 full config 明确没有依赖 agent 的 stop callback。

配置 key 分两层：稳定项目域字段（如 budget.accepted_groups）与固定版本的 verl adapter mapping。未知 key、默认值漂移或 `+field` 未被 dataclass 消费必须报错；保存 resolved Hydra config 与 mapping version，禁止“忽略未知参数继续跑”。

bulk root 默认 `/data/WSH/medical-post-train-artifacts`；run 真实路径由环境配置控制并记录 resolved absolute path。Git 留 compact index 和 hash，bulk 数据不进 Git。model cache、preprocessed data、checkpoints 与 code 分开，避免 git clean 波及 active run。

## 实现顺序与工作包

1. Runtime/schema/ledger、配置编译、data manifest、CPU verifier 基础；先完成负例验收测试。
2. 数据适配器与 tokenizer/collator；锁 held-out exclusion index 后才产生 20k SFT。
3. SFT + save/reload/resume；同时小范围验证 RL stack 的可行性，避免 SFT 完成后才发现 RL 依赖无解。
4. parser/reward/rollout；同一接口交 Stage 3/4，避免重复实现。
5. group refill + verl controller；CPU 合成数据验证，然后真实 smoke/pilot/full。
6. evaluation、case mining、serving；性能 benchmark 由独立 run controller 执行。

## 测试策略

CPU：CSV 多行/role/schema、dedup option permutation、assistant mask、parser 歧义、奖励 gating 上下界、16 种 G=4 binary pattern、group_id 冲突、过滤与 total reward 解耦、budget 达标/短跑拒绝、事务重放/断尾日志、receipt hash 篡改。

真实 GPU：Qwen LoRA 梯度与重载、8B token-length memory sweep（有界 smoke）、FSDP2 single rank、vLLM adapter A/B 输出或 logits 区分、sleep/wake 两轮、old/new logprob/mask parity、GSPO 数值与梯度对照、kill/resume 后 optimizer/参数/采样计数一致。

验收：假造 40-step final adapter、5000 generated 但不足 accepted、pilots 合并冒充 full、遗漏 rejected token、漏模型/漏 difficulty 输出、仅服务启动均必须失败。无需写验证 Markdown 改字这类低价值测试；本次 planning checker 重点保护合同和状态。
