# 未决问题与关闭路径

这些问题不阻止本次规划提交，但相应P0未关闭前不启动正式受影响run。当前没有要求owner立即答复的问题，也没有自动降低研究预算。

| ID | 待决定事项 | 已知 / 缺失 | 下一次最小行动 | gate / 决策人 |
|---|---|---|---|---|
| Q01 | 选定runtime能否在5880 Ada单卡工作 | metadata可解析不是ABI验证；FA2未安装 | 独立环境import/FA2/sleep/LoRA两轮探测，核NCCL需求 | S0→S1/S4；实现工程师 |
| Q02 | 13.0.2 toolchain的实际安装路径或container digest | 系统nvcc12.0；未拉容器 | 查可用container runtime，选私有工具链/镜像并固定hash | 依赖freeze；实现工程师 |
| Q03 | 8B BF16 LoRA真实峰值与速率 | 仅形状估计，无模型实测 | 有界sequence sweep和1-2次真实backward | S1 full前；实现工程师 |
| Q04 | verl v0.9.0 + recipe adaptation的单卡切换方式 | 原生支持路径存在，组合未测 | world_size1、actor offload→rollout sleep、GPU timeline | S4；实现工程师 |
| Q05 | 中文semantic encoder | MedEmbed en、BGE-M3 multilingual，均无本地质量结果 | train-only64对与否定/数字反例；保留失败 | Stage2 reward freeze；D-003 |
| Q06 | response512是否保留 | Qwen thinking有长输出风险，无截断率 | 32prompt长度对照；共享变更记录，重算hours | S2/S4 generation freeze；D-004 |
| Q07 | 清洁SFT每源10k+val是否足够 | 字段已核，未下载清洗 | 全源去重后quota与length统计 | Stage1；预算不足须owner合同decision |
| Q08 | CMExam难度值方向/缺失率；CMB题与答案版本 | 只读header与card，没有全量审计 | 数据准备验证annotations语义及答案join，test分数不可见 | S5协议freeze；实现工程师 |
| Q09 | 实际mixed acceptance是否使5k预算可承受 | 0.35仅工作假设，可能趋零 | Stage2正式profiling+Stage3真实refill | S4资源计划；预算改变需owner |
| Q10 | 第二份durable副本/外部artifact store | `/data`本地挂载存在，无备份验证 | 选第二物理盘目录+hash/restore；若需要外部存储再确定目的地 | 长run前；本地方案实现工程师，外部目标待owner |
| Q11 | checkpoint adapter-only + optimizer恢复支持细节 | 上游实现存在，没跑 | 两种crash时点+optimizer state roundtrip | S4 formal前；实现工程师 |
| Q12 | CPU语义奖励会否成为pipeline瓶颈 | 38CPU核，但tokenizer/chunk耗时未知 | 单线程与8线程批量计时，限定Ray线程 | Stage2；实现工程师 |
| Q13 | 单训练seed是否足够项目结论 | 原合同未要求多seed，主比较保留seed42 | 完成mandatory后按CI/资源提额外seed，不能挤占baseline | S5局限；新研究proposal |

未来如Q03证明BF16不可行，或清洗后Q07/Q08无法满足quota，需要明确proposal描述原因、替代、比较影响与owner决定；在此之前状态BLOCKED，不将少量smoke当full。Questions更新保留已关闭证据、日期与decision链接，不覆盖原未知状态。

## Stage 0 回答与仍需研究的事项

已用真实运行回答：本机可加载BF16 Qwen3-8B并对r32 LoRA反传到2048；独立固定runtime可import并运行nativeFSDP2/vLLM；受控完整LoRA optimizer恢复通过。默认LoRA重复数值不稳定，最终采用native batch invariance，并完成3次sleep和真实adapter更新验证。

仍OPEN：两种semantic encoder都不能排除剂量反例；1024输出仍不达闭合阈值；自然mixed acceptance、跨引擎matched-temperature logprob parity、完整Ray controller、正式数据去重/配额、外部备份与主机重启恢复。mini=4和共享更长response均仅proposal，未改变formal合同。本轮这些研究问题不要求owner即时批准，也不授权开始Stage1 FULL。
