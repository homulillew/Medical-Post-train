# RTX 5880 Ada 算力与存储预算

**这里全部 GPU-hour 数字都是分析情景，不是实测 benchmark，也不是确定性时间承诺。** 本次只测过64×64 BF16张量；没有8B实际token统计、吞吐或mixed acceptance。硬件容量已核实，因此排除了套用A100/H100速度的做法；吞吐仍须后续probe/pilot校准。名为realistic的列是工作假设，不是统计意义的期望值。

## 显存账本

依据固定Qwen配置：36 layers、hidden4096、intermediate12288、32 query heads、8 KV heads、head_dim128、vocab151936、untied embeddings。按dense权重形状估算约8.19B参数，BF16 base约15.26 GiB。r32在7类linear层约87.29M可训练参数；FP32 adapter+grad+Adam m/v约1.30 GiB（以实际dtype和optimizer实现复核）。

| 活动阶段 | 估算常驻与临时需求 | 实机规划 |
|---|---|---|
| SFT | base15.26 + LoRA/optim1.30 + activations约6–14 + logits/workspace约2–5 GiB | 大致25–36 GiB；2k seq、microbatch1、checkpointing先验证，保留>=3 GiB余量 |
| 独占rollout | base15.26 + LoRA + KV + engine/context/workspace | max_num_seqs16、总context2048；不要直接使用默认超长context |
| actor old/backward | base+LoRA/optim + one microbatch activations | vLLM先sleep；entropy/logprob用chunk避免vocab logits瞬时膨胀 |
| 同时双base | 约30.52 GiB，仅权重，不含KV/激活 | 不能据此说48GB足够，初始化与切换峰值必须测 |

BF16 KV每token理论量：`2(K,V)*36*8*128*2 bytes =147456 bytes=144 KiB`。16条×2048完整context上限约4.5 GiB，32条约9 GiB，此外还有碎片/工作区。这解释了为什么32 accepted trajectories可以拆成最多16条并发rollout、actor按microbatch累积，不能将batch大小等同于同时驻显存数。

原生LoRA多进程sleep/offload使host RAM需要两份base、optimizer、checkpoint临时序列化、数据与Ray object store。预算约60–85 GiB，监控available RAM下限16 GiB；限制CPU embedding与Ray并发，禁止无上限worker。`use_shm`默认关，避免把全部base复制到tmpfs再形成重复占用。

## 时间模型

- SFT：`20,000 * mean_total_tokens / effective_training_tokens_per_second`。
- Vanilla生成：`5,000*4*mean_response_tokens / aggregate_decode_tokens_per_second`。
- Dynamic生成：上述生成量除以当前平均mixed acceptance a；额外overflow/retry另外计。
- 每组RL actor+old：`20,000*(mean_prompt+mean_response)*(1/actor_tps + 1/old_tps)`。
- 两组每组625次更新；切换时间每更新另计，validation为11×512 + 至多3×1024 =8704 responses/变体。
- 最终eval：3×(6811+2000)=26433，再加50×3 open-ended。
- 末尾overhead因子覆盖checkpoint/CPU reward/prefill波动/调度；这些不是无限容量，真实reward CPU若超预算必须重估。

| 假设输入 | best-case | realistic工作假设 | adverse有限情景 |
|---|---:|---:|---:|
| SFT平均总tokens/例 | 512 | 1024 | 2048 |
| SFT有效训练tokens/s | 1000 | 450 | 150 |
| RL平均prompt/output tokens | 256/128 | 384/256 | 768/512 |
| rollout aggregate output tokens/s | 300 | 160 | 60 |
| actor有效total tokens/s | 800 | 350 | 100 |
| old-logprob total tokens/s | 1500 | 700 | 200 |
| Dynamic acceptance a | 0.65 | 0.35 | 0.10 |
| 每update切换秒数 | 5 | 15 | 60 |
| 调度/保存等余量倍数 | 1.10 | 1.20 | 1.35 |

这些吞吐输入都是情景参数，不是已发表或本机测得的RTX5880基准值。特别是LoRA batch=16的aggregate decode不等同于单请求tokens/s。

## 主线预计GPU占用小时

“GPU占用小时”指任务占有该单卡的wall time（含CPU reward等待/切换），不是按NVML utilization积分；额外报告active kernel time时另名。

| 工作 | best-case h | realistic h | adverse h |
|---|---:|---:|---:|
| Stage1：20k完整SFT | 3.13 | 15.17 | 102.40 |
| Stage2：1000×4 profiling | 0.52 | 2.13 | 12.80 |
| Stage3：256 accepted real refill | 0.21 | 1.56 | 32.77 |
| Vanilla GSPO：5000组，含其validation/selection | 9.19 | 36.72 | 249.92 |
| Dynamic GSPO：5000 mixed组，含其validation/selection | 10.60 | 56.53 | 825.92 |
| Stage5：三模型完整eval + open-ended | 3.47 | 14.18 | 85.07 |
| Stage6：consistency + benchmark，请求级调度预留 | 0.50 | 2.00 | 8.00 |
| 合计 | **27.61** | **128.29** | **1316.87** |

算式输入与未四舍五入输出见 [compute-scenarios.json](evidence/compute-scenarios.json)。Stage6的并发1不适用batch aggregate吞吐，所以表中单独采用0.5 / 2 / 8 h的请求级调度预留（仍是假设）；初版仅按aggregate吞吐计算的更低值在JSON中保留为not_adopted，不当成测量。最终用校准后的请求级耗时替换。

realistic情景下Vanilla生成约5.12M output tokens，Dynamic约14.63M（未加overflow）；两组接受的20k trajectories数量相同。一次run不能靠“总共625步”隐去约2.86倍生成放大。Stage4 validation另外每变体约2.23M tokens，成本不可忽略。

严格worst-case**无有限上界**：a趋近0、refill starvation、机器故障或不兼容可使合同无法完成。adverse只是a=0.1、长度饱和且吞吐低的有限场景；这应触发诊断/资源讨论，而不是自动跑数周或降低5000组。`a = 1-p^4-(1-p)^4`仅在同题独立同分布且固定正确率p的模型中成立；真实组相关、题难度异质，不能用平均accuracy代入当实测acceptance。

## 后续校准与探索预算

本次不做以下训练；列出后续上限与用途：

| 验证 | 拟定小预算 | 解决的问题 |
|---|---|---|
| 8B SFT memory sweep | 512/1024/2048 tokens各最多2个microbatch；整体<=15min | BF16/LoRA真实峰值；超时保留诊断，不升级为pilot |
| vLLM/actor切换 | 固定8 prompts×4，2个policy同步周期，<=20min | 单卡共存、cold/warm load、adapter真实性 |
| MedEmbed中文有效性 | 64对train-only text，CPU优先；<=15min/encoder | 语义区分、chunk/UNK、CPU服务耗时 |
| length诊断 | 32 train prompts×4，比512/1024，max一次<=20min | `<answer>`闭合率、截断、token成本；扩2048另记decision |
| SFT pilot | 1024例/64 updates | 粗略按formal约5.12%训练量，另有启动/验证 |
| RL pilots | 每变体512 groups（formal的10.24%） | 约best 2h / realistic 10h / adverse>100h；adverse时先停诊断 |

先测有效tokens/s（SFT与actor是total processed，rollout是output aggregate），峰值显存、cold load、switch time、CPU reward、acceptance与长度分布，再重算预算。探测不是Stage done；optional诊断总GPU上限初始2h，pilot另计。约128h主线+10–15h pilot/诊断仅是当前工作预算，可能需数个会话与多天持久调度。

## 正式参数建议

- **20k SFT、15k pool、5000 training groups、G=4保持。** 当前证据不足以证明必须降低。
- 保留r32；估计LoRA状态相对base小，首选减microbatch/关闭graphs/正确offload。
- response512是待验证风险；如果闭合answer率<95%或截断率>5%（train/val diagnostic），**proposal** 同时提高两组到1024，必要时2048，重新测预算和primary generation config。阈值是诊断触发条件，不是删掉难题的规则。
- 缺内存时先顺序切换再考虑架构；BF16→quantized LoRA是project-contract proposal，不自动采用。
- acceptance<0.1持续3个refill窗口时暂停并保存case；提temperature/shared length/组大小探索，不自动替换baseline或生成永久黑名单。

## 磁盘与保留

预留300 GiB工作额度+100 GiB安全余量：base单份约16.4GB、依赖/toolchain/wheels约20–60GiB、原始/处理数据与token cache约10–30GiB、raw rollout与token/logprob records约5–30GiB、两组checkpoints约30–80GiB（adapter-only权重+optimizer/extra若通过），另留export和故障证据。若只能保存完整base的FSDP checkpoints，每10份可增加约150GiB/run，转为预留>=600GiB并调整checkpoint保留策略，不能删除唯一恢复点。

磁盘当前充足不等于无风险；每次prepare/checkpoint前检查free bytes、临时双写空间、目标mount一致。active run至少保留最近2个有效完整恢复点、所有validation便携adapter、最终与选中checkpoint及失败证据。每次关键checkpoint复制到第二个物理盘是本机冗余；外部备份目的地尚未确认，见OPEN_QUESTIONS。
