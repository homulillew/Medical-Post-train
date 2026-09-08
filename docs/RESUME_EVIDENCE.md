# Resume Evidence Ledger

本表只覆盖Stage1已通过全局验收的实际产物。[Stage1 verifier](../experiments/stage1/verification-final.json) 已PASS，10个gate均通过；下列声明在所列范围内为VERIFIED。临床正确率和考试test结果没有在本阶段测量。

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
| 提出了GSPO或完成Dynamic Sampling训练、提升X% | Stage2–6未开始，无任何正式RL结果 | 完成相应阶段的全预算、对照与评价 |
| 分布式FSDP训练与正式线上服务性能 | 本轮是单GPU自定义SFT与离线生成 | 后续真实训练/部署/请求级benchmark |
| 整个项目实测只需124.52小时 | 该数字是含Stage1实测的条件性未来预算，大多数项仍是假设 | 后续每阶段实际完整计时 |

失败记录、案例与未知量属于证据，不能为了简历措辞删掉。Base没有answer标签也不等于没有语义答案；格式100%不写成正确率100%。
