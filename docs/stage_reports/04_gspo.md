# Stage 4 — Online GSPO, Dynamic Sampling, and Controlled Ablation

Status: **FULL_PASS**, pending genuine human review and final acceptance.
`READY_FOR_STAGE5=NO`; Stage5 remains `NOT_STARTED`.

Scientific narrative: ChatGPT text and conclusions supplied by the project owner.
Repository wiring and evidence checks: Codex. This document does not record a
completed human review or a full Stage4 acceptance PASS.

Formal runs:

- Vanilla: `s4_formal_vanilla_20260909T155928_24aec5`
- Dynamic: `s4_formal_dynamic_20260909T155929_18b61a`

The [documentation handoff](../../experiments/handoffs/stage4_final_docs_to_chatgpt_v1.json)
retains exact source paths and SHA256 references. The
[input audit](../../experiments/stage4/documentation_input_audit_v1.json)
checks the owner-provided numerical claims against that evidence.

## 1. Research question

在单张约 48 GB GPU 的 Online RL 预算下，correctness-aware Dynamic Sampling
如何改变进入 optimizer 的训练信号组成、equal-update learning behavior、rollout
cost 和 held-out generalization behavior？

Dynamic Sampling 是 update-allocation intervention。它改变哪些 prompt groups
进入更新，不修改 reward 或 optimizer。Vanilla 与 Dynamic 使用相同的主要配置，
预期科研差异只有 `sampling_mode`。验证集上的表现、训练信号和生成成本分别报告。

## 2. System setup

| Component | Frozen implementation |
|---|---|
| Base model | Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Precision | BF16 base; float32 LoRA parameters |
| LoRA | r=32, alpha=64 |
| Targets | q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj |
| GPU | One NVIDIA RTX 5880 Ada Generation; 46068 MiB visible VRAM |
| Actor | Native verl 0.9.0 FSDP2 actor on one GPU |
| Rollout | vLLM 0.24.0, native sampling and batch-invariant path |
| Other runtime | torch 2.11.0; Transformers 5.5.3; PEFT 0.18.1 |

所有 RL 条件从同一份 verified Stage1 final adapter 初始化：
`s1_formal_20260908T152119_60040b`，adapter SHA256
`1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64`。
这是单卡实现，不是多 GPU 训练。

同一张 GPU 交替承担 rollout 与 actor 更新。vLLM 在 actor 运行前 sleep，
actor 退出并释放显存后，vLLM wake，再同步新 adapter。原始计时分别记录
sleep、wake、actor、generation、reward 和 sync；它们不代表多卡并行吞吐。

Sources: [SFT initialization](../../experiments/stage1/initialization_manifest.json),
[formal pair](../../experiments/stage4/formal_pair.json),
[rollout lifecycle](../../src/medical_posttrain/rl/online.py).

## 3. GRPO-style advantage + GSPO loss

本项目使用 **GRPO-style group-relative outcome advantage estimator + GSPO
sequence-level policy objective**。前者决定组内 credit assignment，后者决定
如何用 importance ratio 和 clipping 更新策略。

每个 prompt 生成 G=4 条 trajectories。GRPO-style estimator 在组内对 outcome
reward 做中心化和标准差归一化，得到每条 trajectory 的 advantage，因此不需要
额外 critic。GSPO 使用 sequence-level importance ratio：

`ratio_i = exp(mean_t(log pi_current(y_it | context) - log pi_old(y_it | context)))`

这个 ratio 是逐 token 概率比的几何平均。它在 sequence 层面接受 clipping，
与最终答案 correctness 为主的 trajectory-level reward 相匹配。这是设计理由；
本项目没有 GRPO-loss vs GSPO-loss 的 formal ablation，不能声称 GSPO 已被实验证明更优。

每个 window 有 8 groups、32 trajectories，分成两个 4-prompt minibatches。
全部 32 条 trajectory 的 actor old logprobs 在两个 optimizer minibatch 之前冻结。
old policy 是本窗口采样对应的策略；系统没有 permanent reference worker。
冻结配置为 no critic、no KL reward/loss、LR=1e-6、clip low/high = 3e-4 / 4e-4、
ppo_epochs=1、grad clip=1、entropy coefficient=0。

两个条件共享 correctness-gated hybrid reward：
`R = 0.8 * acc + 0.15 * acc * semantic + 0.05 * format`。
错误答案不能靠 semantic similarity 获得正确性门控的语义奖励。Dynamic eligibility
只读取 binary correctness 是否在组内形成对比，不读取总 hybrid reward。

Sources: [native actor](../../src/medical_posttrain/rl/actor.py),
[reward manifest](../../experiments/stage2/reward_manifest.json),
[frozen shared config](../../configs/stages/s4_formal_shared.json).

## 4. Why LR=1e-6

优化诊断在同一份保存的 8×4 batch 上比较三个 LR；每个条件使用 fresh optimizer。
较大 LR 出现明显 ratio drift，许多 ratio 超过 tight clip 边界。实际 objective
clipping 还取决于 advantage 的符号，不能把所有越界都算成被裁剪的更新。

| LR | Second-mini ratio min / max | Objective clip fraction | Any-bound exceedance |
|---|---|---:|---:|
| 1e-5 | 0.698170 / 1.005568 | 0.1875 | 0.8125 |
| 3e-6 | 0.933328 / 1.006385 | 0.1875 | 0.9375 |
| 1e-6 | 0.994234 / 1.010573 | 0.2500 | 0.9375 |

1e-6 避免了这批数据中早期观察到的极端 ratio drift，clipping 仍实际 active。
该选择依据是优化行为，不是测试集或验证集成绩。tight clip 阈值本身不是 bug；
也不能从上表声称 objective clipping 随 LR 单调增大。

初始诊断 `s4_optimization_diagnostic_20260909T065945_66b040` 在 optimizer 更新前
因配置序列化失败。修复后的 `s4_optimization_diagnostic_20260909T070249_a3a605`
复用了该次真实 batch，保留原始失败及生成成本，没有用新采样替换不利输出。

两次 full run 的 first minibatch ratio 为 1、clip 为 0，符合 old=current sanity。
second minibatch 出现真实 policy drift。完整运行记录中没有 NaN/Inf，
没有 optimizer inactive windows；entropy 和 response length 也未见明显 collapse。
这些是本次完整运行的诊断结果，不把一个 minibatch 的稳定性推广到任意配置。

Sources: [optimization decisions](../implementation/STAGE4_DECISIONS.md),
[diagnostic verification](../../experiments/stage4/diagnostic_verification.json),
[full-run health statistics](../../experiments/stage4/postformal_analysis_v1.json).

## 5. Formal controlled comparison

| Variant | Training groups | Policy windows | Optimizer steps |
|---|---:|---:|---:|
| Vanilla | 5000 | 625 | 1250 |
| Dynamic | 5000 accepted mixed groups | 625 | 1250 |

每个条件实际完成 20,000 条 training trajectories。两者保持相同的 SFT initialization、
reward、G、optimizer、LR、GSPO、decoding、parser 和 validation protocol。
逐项配置比较只发现 `sampling_mode` 不同。共同 rollout 参数包括 temperature=0.6、
top_p=1 和 max_response=1024。

Dynamic 拒绝 all-correct 和 all-wrong groups，并继续生成候选直到获得本窗口所需
mixed groups。已经生成但超出窗口容量的 mixed groups 标为 `overflow_eligible`，
保留其成本和原始输出。正式预算只统计进入训练的 groups。

Sources: [formal pair and confounders](../../experiments/stage4/formal_pair.json),
[Vanilla raw PASS](../../experiments/stage4/vanilla_formal_verification.json),
[Dynamic raw PASS](../../experiments/stage4/dynamic_formal_verification.json).

## 6. Equal-update results

以下结果来自固定 monitor512，均不是 final test。

| Checkpoint | Monitor accuracy |
|---|---:|
| SFT initialization | 55.078125% |
| Vanilla5000 | 67.578125% |
| Dynamic5000 | 68.5546875% |

最终 Dynamic−Vanilla 为 **+0.9765625 pp**。配对结果为 corrected 44、regressed 39；
exact McNemar p=0.660884（约 0.661）。paired bootstrap 95% CI 为
**[−2.5390625, +4.4921875] pp**，约 [−2.54, +4.49] pp。
Final monitor does not establish statistically credible Dynamic superiority.

在全部预注册正数 equal-update milestones 上，Dynamic accuracy point estimate
均高于 Vanilla。最大测得 gap 为 **+4.1015625 pp @4096 groups**，不是仅在训练早期领先。
上述曲线描述一次固定训练种子的结果；题目级 bootstrap 不估计训练种子之间的变异。

Sources: [all update milestones and paired statistics](../../experiments/stage4/postformal_analysis_v1.json),
[curve data](../../experiments/stage4/postformal_data_v1/update_budget.csv).

## 7. Signal-density result

对两个正式 run 保存的 native advantages、mask、reward 和 update artifacts 回放，
完整统计双方各 5000 training groups。nonzero 阈值预先固定为 `|A| > 1e-6`。
correctness-discriminative group 同时包含正确和错误轨迹，且满足
`mean A(correct) > mean A(wrong)`。

| Density among training groups | Vanilla | Dynamic |
|---|---:|---:|
| Correctness-contrast | 41.02% | 100% |
| Correctness-discriminative | 41.02% | 100% |
| Any-nonzero-advantage | 75.7% | 100% |

Dynamic 将具有明确 correct-vs-wrong advantage separation 的 training-group
比例从 **41.02% 提升到 100%**。correctness-discriminative group-density ratio
约 **2.4378×**。这是训练组信号组成的变化，不能表述为 gradient quality 提升 2.44 倍。

Vanilla all-correct groups 中约 **84.65%** 仍有 nonzero advantage。
hybrid reward 的 semantic/format shaping 可以产生组内差异，因此 nonmixed group
不等于零梯度或无训练信号。保存的 artifacts 不支持 per-group gradient norm attribution；
该字段保持 `NOT_IDENTIFIABLE_FROM_RETAINED_ARTIFACTS`。

Sources: [signal-density analysis](../../experiments/stage4/signal_density_analysis_v1.json),
[native replay verification](../../experiments/stage4/signal_density_verification_v1.json).

## 8. Rollout cost

| Variant | Generated groups | Selected training groups | Physical prompt + output tokens | Mean output tokens / trajectory |
|---|---:|---:|---:|---:|
| Vanilla | 5000 | 5000 | 5,964,478 | 264.59 |
| Dynamic | 14,872 | 5000 mixed | 17,443,344 | 259.57 |

Dynamic group amplification 为 **2.9744×**，rollout token ratio 约 **2.92454×**。
平均 output length 没有增长；额外成本主要来自 refill 和更多 generated candidate groups，
不是 verbosity explosion。该 token 口径包含被拒绝和 overflow 的输出，prompt 每次
encounter 计一次；validation/control generation 单独核算。

Dynamic is update-efficient under an equal training-group budget, but is not
demonstrated to be rollout-token-efficient. 这里的 update-efficient 指本次固定
equal-update milestones 的 accuracy point estimates，不表示最终差异已达到统计显著。

Source: [cost accounting and exclusions](../../experiments/stage4/postformal_analysis_v1.json).

## 9. Equal-token comparison

使用已有 shared 1M–5M token thresholds，各阈值取首次跨越该预算的 committed
checkpoint。保留实际 token overshoot，不插值构造不存在的 checkpoint。
Dynamic 在 5 个 shared thresholds 中仅在 **4M** 点领先，没有稳定的 equal-token 优势。

Equal-update advantage cannot be re-labeled as equal-token compute efficiency.
2.92454× 是 rollout token ratio，不能直接替换为 GPU-hour ratio、费用比例或
end-to-end compute ratio。

Source: [shared-token curve and actual crossings](../../experiments/stage4/postformal_data_v1/shared_token_budget.csv).

## 10. Frontier Diagnostic

该 diagnostic 使用 exactly 1000 isolated validation prompts，对 SFT、Vanilla5000、
Dynamic5000 各生成 G4，共 **12,000 responses**。optimizer updates=0，raw replay PASS。
数据与 monitor、selection、RL、SFT 和 sealed test clusters 的重叠均为零。

Vanilla trajectory accuracy 为 **67.3%**，Dynamic 为 **68.6%**，差值 +1.3 pp。
exploratory paired 95% CI 为 [−0.025, +2.675] pp，跨过 0。

预注册 H1：SFT mixed prompts are more often resolved to all-correct by Dynamic
than Vanilla。结果为 **NOT SUPPORTED**。
在 SFT-mixed 的 517 prompts 中，Vanilla 将 264 题变为 all-correct（约 **51.06%**），
Dynamic 为 261 题（约 **50.48%**）；trajectory accuracy 基本相同。
不能声称 Dynamic 通过直接解决这批 held-out mixed frontier prompts 获益。

Sources: [Frontier analysis and H1–H4](../../experiments/stage4/frontier_diagnostic_v1/analysis.json),
[isolation manifest](../../experiments/stage4/frontier_diagnostic_v1/manifest.json),
[raw replay](../../experiments/stage4/frontier_diagnostic_v1/verification.json).

## 11. Hard-tail finding

在 **SFT-G4-all-wrong hard-tail proxy** 中，n=231 prompts。
Vanilla trajectory accuracy 约 **24.13%**，Dynamic 约 **29.33%**；差值约 +5.19 pp，
exploratory paired bootstrap 95% CI 约 **[+2.16, +8.33] pp**。

This exploratory result suggests Dynamic's extra benefit may concentrate on a
harder tail rather than directly resolving the same static SFT-mixed class.
G4 classification 存在 sampling noise，all-wrong 不等于绝对难题。该结果来自
validation subgroup，并非训练种子推断，也没有证明因果解释。

Source: [fixed SFT-defined subgroups](../../experiments/stage4/frontier_diagnostic_v1/analysis.json).

## 12. Random-3× causal control

辅助 control 从原始 SFT fresh start，完成 512 training groups、64 windows、
128 optimizer steps。它没有替换或重跑正式对照。

| Condition | Generated groups | Training groups | Rollout tokens | Correctness-discriminative selected density |
|---|---:|---:|---:|---:|
| Vanilla512 | 512 | 512 | 615,203 | 48.2421875% |
| Dynamic512 | 1,360 | 512 | 1,633,072 | 100% |
| Random3x512 | 1,360 | 512 | 1,627,034 | 44.53125% |

Random3x 精确匹配 Dynamic first64 windows 的 generated-group schedule，包括
候选 prompt、encounter 顺序和生成数量，但使用自身策略生成 fresh responses。
它不是 exact token-matched；两者 rollout tokens 相差约 0.37%（以 Dynamic 为分母）。
记录的 active phases 合计约 **3.03 h（Dynamic）与 3.06 h（Random3x）**，
这一口径不包含所有外层加载/校验时间，也不把嵌套 actor 时间重复相加。

选择在生成前冻结：对 `20260914:window_index:encounter_index:prompt_id` 做 SHA256，
按 hash 升序选前 8 组。不读取 correctness、reward、parser、semantic、format 或
output length。生成数量不随新策略结果变化；valid wrong responses 不重采。

Fresh isolated ablation validation512，统一 greedy n=1：

| Checkpoint | Accuracy |
|---|---:|
| SFT | 59.9609375% |
| Vanilla512 | 63.8671875% |
| Random3x512 | 62.5% |
| Dynamic512 | 64.6484375% |

Dynamic−Random3x 为 **+2.1484375 pp**；43 W→C、32 C→W，exact McNemar
p=0.248046（约 0.248），bootstrap 95% CI **[−1.171875, +5.46875] pp**。
Random3x−Vanilla 为 **−1.3671875 pp**，95% CI [−5.078125, +2.1484375] pp，也跨过 0。

Point estimates are more consistent with a correctness-aware selection effect
than with an exposure-only explanation. Single-seed uncertainty leaves the causal
mechanism **INCONCLUSIVE**. 不能升级为 SUPPORTED，也不能从不显著推断两者等价。

Sources: [preregistration](../../experiments/stage5/aux_random3x_protocol_v2.json),
[training comparison](../../experiments/stage5/aux_random3x_training_comparison_v1.json),
[auxiliary evaluation](../../experiments/stage5/aux_random3x_eval_analysis_v1.json),
[full raw verification](../../experiments/stage5/aux_random3x_verification_v1.json).

## 13. Negative result retention

以下结果与正向点估计一起保留：

1. Dynamic final monitor superiority 未达到统计显著。
2. Equal-token comparison 没有稳定优势。
3. Frontier mixed-resolution H1 未获支持。
4. Random3x selection-effect CI 跨过 0。
5. Dynamic 存在 item-level regressions，不能只展示纠正案例。
6. Single training seed 是重要限制，题目级 bootstrap 不能替代多种子实验。

审阅包中的 `S4-MR-02` 展示 Dynamic 正确、其余三个条件错误的例子。
`S4-MR-03` 展示 SFT/Vanilla 正确而 Dynamic/Random3x 错误的回退。
两个案例均来自辅助验证集，不能代表总体胜率或临床有效性。
它们的完整回答已提供，但 **人工审阅尚未确认**；拟议 observation 仍是 draft。

Sources: [full-response review packet](../../experiments/stage4/manual_review_packet_v1.md),
[unconfirmed proposed reviews](../../experiments/stage4/formal_manual_cases_draft_v1.json).

## 14. Online RL transaction recovery

正式 Vanilla 事故发生时，controller 最后提交的状态为 cursor=3216、
policy windows=402、optimizer steps=804。window0402 的 actor 实际已完成
optimizer steps **805、806**，并写入 durable checkpoint，随后正常退出。
GPU-release assertion 在 controller sync/commit 之前失败。

如果从旧 controller 状态简单 replay，会重复已经完成的 805/806。
恢复采用完整 checkpoint：检查 marker、optimizer/scheduler/RNG 和参数身份，
reload 后同步 adapter，再提交 controller。cursor 前进到 **3224**，optimizer
step=**806**，replayed optimizer steps=**0**。下一 fresh window 提交到
**3232 groups / step808，PASS**。

Rollout/update/checkpoint/controller-commit was treated as a transaction boundary.
系统保留生成和更新事实，再根据 durable checkpoint 判断采用还是回滚，避免计数器
与参数状态脱节。GPU release guard 在原阈值下有限等待真实 NVML 读数，不伪造显存值。
原事故没有保存当时的具体 NVML 数值；延迟 teardown 是符合证据的解释，不能据此
断言特定驱动缺陷或精确释放延迟。

Random3x 提供另一个真实 fresh-process resume：在 32 groups / step8 终止旧进程，
确认旧 PID 已退出，新进程恢复 optimizer、scheduler、RNG 和下一采样位置，
下次提交到 step10，`optimizer_replayed=false`。

Sources: [recovery evidence bundle](../../experiments/stage4/recovery_interview_case_v1.json),
[formal adoption PASS](../../experiments/stage4/vanilla_formal_resume_verified_002.json),
[next fresh window PASS](../../experiments/stage4/vanilla_formal_fresh_guard_verified_002.json),
[Random3x real resume](../../experiments/stage5/aux_random3x_resume_v1.json).

## 15. vLLM numerical case

Stage0 在 default LoRA rollout path 发现 batch-related numerical instability。
最终冻结 `VLLM_BATCH_INVARIANT=1`、V1/native sampler、LoRA shrink `split_k=1`。
原始失败与接受配置的对比均保留。

这条配置保障了后续采用的数值路径；它没有证明跨进程随机采样逐 token 完全一致。
本项目也没有完成 Stage6 concurrency benchmark，不能把运行兼容性结果当成 serving
并发性能结论。

Sources: [Stage0 runtime evidence](00_runtime_compatibility.md),
[frozen rollout implementation](../../src/medical_posttrain/rl/rollout.py).

## 16. Supported conclusions and acceptance handoff

### Supported conclusions

- GSPO post-training produces strong validation gains over SFT in the completed runs.
- Dynamic changes optimizer signal composition and raises correctness-discriminative group density.
- Dynamic shows persistent positive equal-update point estimates in this formal pair.
- Dynamic pays substantial rollout refill cost.
- Random3x does not reproduce Dynamic's point estimate merely by matching candidate generation budget.
- The causal mechanism remains statistically **INCONCLUSIVE**.
- Transaction-safe recovery was exercised in real runs.

### Unsupported conclusions

- Dynamic statistically significantly beats Vanilla at final monitor.
- Dynamic is globally compute efficient or saves rollout compute.
- GSPO is experimentally proven better than GRPO.
- Dynamic's causal selection mechanism is proven.
- Frontier mixed-resolution H1 is supported.
- Medical clinical validity or final held-out test superiority.
- A 2.4378× group-density ratio means the same increase in gradient quality.

### Acceptance status

| Requirement | Current evidence |
|---|---|
| Both mandatory full budgets | PASS: 5000 groups / 625 windows / 1250 steps each |
| Matched initialization/config, raw updates and lineage | PASS: existing formal raw receipts |
| Final native reload | PASS for both formal runs |
| Real recovery and auxiliary resume | PASS: retained native reload and next-commit evidence |
| Formal curves, signal-density, Frontier, Random3x | Verified artifacts retained; auxiliary findings do not replace formal endpoints |
| Stage report and interview story | Owner-provided narrative wired; awaiting human review/acceptance |
| At least two genuine final manual reviews | PENDING: `manual_reviews=[]` |
| Final deliverables manifest and full Stage4 acceptance | PENDING: draft only; full verifier not run this task |

Exact paths and hash requirements are specified in the
[deliverables schema](../../experiments/stage4/stage4_deliverables_schema_v1.json).
The [draft manifest](../../experiments/stage4/deliverables_draft_v1.json) explicitly keeps
`READY_FOR_STAGE5=NO`. Stage4 stays `FULL_PASS`; it is not `DONE`.

The [Stage5 checkpoint-selection protocol](../../experiments/stage5/checkpoint_selection_protocol_v1.json)
is frozen but has not executed. It retains ten candidates per RL variant and both 5000-group
scientific endpoints. selection1024 and project-model final tests remain untouched.
The next action is genuine full-response review of `S4-MR-02` and `S4-MR-03`, with explicit
confirmation or edits. Only then may final cases/deliverables be prepared and the full verifier run.


## GRPO vs GSPO Auxiliary Objective Ablation

独立单 seed、512-group auxiliary validation 对照已完成。原有 5000-group GSPO formal pair 及其结论保持冻结。本节补充此前缺少的 token-level objective 对照；不是 primary formal GRPO 实验。

GRPO 使用 verl `compute_policy_loss_vanilla`，原生 token ratio 与 dual-clip=3，和 GSPO 同用 native GRPO group-relative advantage、`seq-mean-token-mean` 聚合。六组原始固定批次诊断后按预注册规则选择 LR=1e-06、symmetric clip=0.2，没有使用新 validation 调参。原生 clamp、dual-clip 和 clip 尺度均属于 objective-package 差异，不能把全部差异归因于 ratio geometry。

| Model | Correct/N | Accuracy |
|---|---:|---:|
| sft | 313/512 | 61.1328% |
| vanilla_grpo | 321/512 | 62.6953% |
| vanilla_gspo | 322/512 | 62.8906% |
| dynamic_grpo | 328/512 | 64.0625% |
| dynamic_gspo | 326/512 | 63.6719% |

- `vanilla_gspo_minus_grpo`: +0.1953 pp，95% CI [-2.9297, +3.3203] pp；gain/regress=34/33，exact McNemar p=1。
- `dynamic_gspo_minus_grpo`: -0.3906 pp，95% CI [-3.9062, +3.1250] pp；gain/regress=40/42，exact McNemar p=0.912157。
- `dynamic_minus_vanilla_grpo`: +1.3672 pp，95% CI [-1.9531, +4.6875] pp；gain/regress=42/35，exact McNemar p=0.494382。
- `dynamic_minus_vanilla_gspo`: +0.7812 pp，95% CI [-2.3438, +3.9062] pp；gain/regress=36/32，exact McNemar p=0.716301。

描述性 interaction=-0.5859 pp，95% paired bootstrap CI=[-5.078125, 3.90625] pp。预注册判定为 `NO_CLEAR_OBJECTIVE_DIFFERENCE`；Dynamic 跨 objective 的结果为 `POSITIVE_POINT_ESTIMATE_BOTH_OBJECTIVES`。这是单 seed exploratory validation，非 final test，CI 跨零不代表等效。

Dynamic-GRPO 实际生成 1336 组，amplification=2.6094，rollout tokens=1602795；Dynamic-GSPO 对应 1360 组、2.6562、1633072 tokens。生成量未人为匹配。

四个固定长度区间的 ratio、clipping、absolute surrogate、accuracy 和样本分布见 [raw analysis](../../experiments/stage4/loss_objective_ablation_analysis_v1.json)。Per-length parameter-gradient proxy 为 `NOT_IDENTIFIABLE`；on-policy 长度关联不能解释为长度的因果效应。所有反向结果与无显著差异的比较均保留。

两条 GRPO 均完成真实 32→40 groups / 8→10 steps fresh-process resume 与完整原生 raw verifier。五模型统一评估通过。案例仅为机器候选，未伪造人工审阅。Stage4 仍 `FULL_PASS`，Stage5 仍 `NOT_STARTED`，`READY_FOR_STAGE5=NO`；selection1024 与 final tests 未使用，未执行 full Stage4 verifier。证据入口：[handoff](../../experiments/handoffs/loss_objective_ablation_to_chatgpt_v1.json)。
