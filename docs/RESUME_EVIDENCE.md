# Resume Evidence Ledger

本表按阶段记录实际产物与其验收状态。[Stage1 verifier](../experiments/stage1/verification-final.json) 已PASS，10个gate均通过；下列声明在所列范围内为VERIFIED。临床正确率和考试test结果没有在本阶段测量。

| ID | 候选表述 | 状态 | 证据与范围 | 简历可用 |
| --- | --- | --- | --- | --- |
| S1-C01 | 在单张RTX5880 Ada上完成Qwen3-8B BF16 + r32 LoRA医疗SFT，20,000个唯一样本、一完整epoch、100%覆盖 | VERIFIED | `s1_formal_20260908T152119_60040b`，canonical IDs/metrics、final checkpoint、配置与原始数据hash | YES（限本行范围） |
| S1-C02 | 完整处理9,168,530 total tokens、7,741,165 assistant监督tokens；正式worker计时2.537h、更新吞吐1074.84 total tokens/s、NVML峰值37.188GiB | VERIFIED | 正式summary与1250条raw update；worker与update-phase含义不同，重载/生成另计 | YES（限本行范围） |
| S1-C03 | 实现原生Qwen模板下的assistant-only监督、EOS/PAD边界校验、全21k重tokenize与跨benchmark词法簇隔离 | VERIFIED | `s1_data_20260908T144854_bac3a7`，tokenization_verification、raw manifest、duplicate graph、31项测试；不能保证所有语义改写零泄漏或CoT事实正确 | YES（限本行范围） |
| S1-C04 | 在真实1024例pilot上验证退出进程后的LoRA/Adam/scheduler/RNG/样本cursor恢复，下一步参数与loss对参考误差均0 | VERIFIED | `s1_pilot_20260908T150104_71a69a`，两次physical attempt与resume receipt；不是正式GSPO/FSDP恢复证据 | YES（限本行范围） |
| S1-C05 | 在两源各25条固定验证prompt上，SFT的think/answer闭合与严格格式均50/50，截断0/50；平均输出346.22tokens | VERIFIED | `s1_evaluation_20260908T175526_8cfbc0`，cap1024、temperature0.6/top_p1/top_k-1、seed42；单次采样、无医学准确率结论 | YES（限本行范围） |

最终共同初始化见 [initialization manifest](../experiments/stage1/initialization_manifest.json)，adapter SHA256为`1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64`。阶段报告及30秒/2分钟/深挖故事见 [Stage1报告](stage_reports/01_medical_sft.md)。完整证据入口为 [selected runs](../experiments/stage1/selected_runs.json)。

## 拒绝的表述

| 表述 | 为什么不能使用 | 所需证据 |
| --- | --- | --- |
| 医学准确率提升或临床可靠 | 仅做源内验证loss/格式/长度，已有原始CoT算术错误和生成参考分歧 | 独立医学评价、适当金标准与临床验证 |
| 消除了所有数据泄漏 | 已实现固定词法近重复簇排除，语义改写仍可能漏检 | 更广的语义审计与外部来源分析 |
| empty-think单独导致全部Huatuo推理消失 | 混合SFT前后对照没有隔离来源作用 | 同预算的来源/target消融 |
| 提出了GSPO或完成Dynamic Sampling训练、提升X% | Stage3–6未开始；Stage2只有固定policy rollout，无正式RL训练结果 | 完成相应阶段的全预算、对照与评价 |
| 分布式FSDP训练与正式线上服务性能 | 本轮是单GPU自定义SFT与离线生成 | 后续真实训练/部署/请求级benchmark |
| 整个项目实测只需124.52小时 | 该数字是含Stage1实测的条件性未来预算，大多数项仍是假设 | 后续每阶段实际完整计时 |

失败记录、案例与未知量属于证据，不能为了简历措辞删掉。Base没有answer标签也不等于没有语义答案；格式100%不写成正确率100%。


## Stage 2 — 正式profiling证据

阶段证据状态：**VERIFIED**，Stage 2 = DONE；[最终verifier](../experiments/stage2/verification-final.json) 的8个gate全部PASS。下列声明仅在明确范围内可用。

| ID | 可核对表述 | 证据和范围 |
| --- | --- | --- |
| S2-C01 | 冻结15,000个唯一、确定性抽取的CMExam train RL候选prompt | [data summary](../experiments/stage2/s2_data_20260909T024052_870aba/summary.json)；继承词法去污染，不声称消除所有语义泄漏 |
| S2-C02 | 固定SFT完成1000题×G4=4000条正式响应，轨迹正确率51.775% | [formal summary](../experiments/stage2/s2_formal_20260909T025736_f607d9/summary.json)；clean train抽样、冻结parser/标签，非test或临床准确率 |
| S2-C03 | 实现GT集合正确性、门控语义与格式奖励；全部错误响应semantic贡献0、total≤0.05 | [reward manifest](../experiments/stage2/reward_manifest.json)、完整raw/vector/reward核对；语义相似不等于医学正确 |
| S2-C04 | all-wrong/mixed/all-correct=23.5%/53.1%/23.4%；初始1/P_mixed≈1.8832 | 同一formal summary与1000组明细；倍率是固定policy估计，没有运行Dynamic refill或训练 |
| S2-C05 | 正式输出950,213tokens，均长237.55、P95=365；截断0%，生成调用吞吐182.27tokens/s | [compute calibration](../experiments/stage2/compute_calibration.json)；吞吐不含冷加载和评分，未来训练总GPU时长仍未测量 |
| S2-C06 | 完成200对受控语义诊断和80条完整输出定性复核 | [manual review](../experiments/stage2/manual_review.json)、[semantic summary](../experiments/stage2/s2_semantic_20260909T024652_cb6286/summary.json)；由执行代理阅读，非医生审定，保留否定/数字不敏感和占位参考等负例 |

报告和30秒/2分钟/深挖材料：[Stage 2 report](stage_reports/02_reward_rollout.md)。完整模型能力提升、GSPO收益、临床可靠性仍不可作为已证实简历表述。


## Stage 3 — verified sampling integration evidence

| Claim | Evidence | Conditions | Resume-safe |
| --- | --- | --- | --- |
| 实现基于accuracy的DAPO-style group filtering与真实on-policy refill，固定SFT下新生成496组并恰好接受256个mixed | `s3_formal_20260909T053101_d99393`; `experiments/stage3/formal_raw_verification.json`; `docs/stage_reports/03_dynamic_sampling.md` | G4，固定SFT与奖励；optimizer_updates=0；不是RL训练预算 | YES — final Stage3 receipt PASS |
| 实测sampling amplification1.9375倍，保留拒绝组234,982输出tokens，accepted token fraction50.638% | `experiments/stage3/cost_summary.json`及完整raw batch commits | 已生成成本，不能称节省rollout compute或GPU利用率 | YES — final Stage3 receipt PASS |
| 验证sampler真实SIGTERM/new-process恢复，保留计数、下一批题与成本 | `s3_smoke_20260909T052408_5114dc`; `experiments/stage3/smoke_verification.json` | 提交边界恢复；不宣称所有in-flight/断电场景或逐token随机重生成一致 | YES |

禁止由Stage3写出accuracy提升、GSPO训练完成、临床可靠或零生成开销等结论；Stage4–6仍NOT_STARTED。
