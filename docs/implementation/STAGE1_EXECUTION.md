# Stage 1 执行冻结前计划

2026-09-08；授权范围为完整 Stage 1，Stage 2–6 保持 NOT_STARTED。预算仍为 medical-o1 zh / Huatuo 各 10,000 唯一训练记录，加各 500 独立 validation，一完整 epoch，目标覆盖 100%。

1. `scripts/prepare_stage1.py` 将固定 revision 的原始来源下载到 `/data/WSH/medical-post-train-artifacts/data/stage1/`，保留文件 SHA256 和逻辑行 ID。CMExam 三个完整 split 与 CMB-Exam test 只投影 question 字段建立 exclusion；答案和难度不进入清洗、trainer 或调参进程的输出。先隔离，再选择 SFT。
2. `data/stage1.py` 实现确定性质量规则、question 精确/近重复、全源清洗及 source-balanced split。真实 tokenizer 保留完整最终答案，超 cap 剔除并同源补足。清洁配额不足则 BLOCKED，保留全量拒绝记录，不复制、改源或降低标准。
3. 检查固定 Qwen Jinja 对历史 assistant reasoning 的行为；以实际模板的角色边界建立 labels，测试多轮、CoT、EOS 与 PAD==EOS。禁止 packing。输出 train/val IDs、排除索引、拒绝原因、cluster、token cache 和按源统计。大文件在 bulk，Git 只存哈希、摘要和少量案例。
4. 独立 Stage 1 训练入口使用 BF16 Qwen3-8B 固定 revision，PEFT r32、七个 projections。初始选择 alpha=64，与已验证运行架构一致；正式采用前在 pilot decision 冻结。起始 LR=1e-4、AdamW、cosine、warmup 3%、gradient clip 1、microbatch 1 / accumulation 16、SDPA、gradient checkpointing、max sequence 2048。
5. 真实 smoke 128 例 / 8 updates；独立 fresh-base pilot 1024 例 / 64 updates。pilot 在保存后退出进程，再恢复 optimizer/scheduler/RNG/step/cursor，并用连续参考对照验证实际下一步。pilot loss、validation、tokens/s、显存、格式决定正式 config；任何重大修订保留原实验并另立 decision。
6. formal 在 clean code revision 和数据/config 哈希冻结后独立初始化，以全部 20k ID 消费完毕作为停止条件。每 100 updates 或 15 分钟安全边界原子 checkpoint，保存最新、前一个和 final；记录完整 coverage 与监督 token 计数。独立后台进程不依赖交互连接，异常保留状态并恢复同一 run。
7. final 新进程 reload、全 validation loss、固定 50–100 条 validation 的 Base/SFT 同 prompt 生成比较，测 think/answer 闭合、长度与异常；必要时对 1024/2048 作共享长度建议。仅为 Stage 1 heldout sanity，不运行考试 test evaluation。
8. `scripts/verify_stage.py --stage 1` 从原始指标/ID/checkpoint/恢复证据核对预算与完整性，成功后才允许 VERIFIED；报告、案例、面试材料和简历证据齐全后 DONE。旧 Stage 0 历史收据保留，不冒充 Stage 1 验收。

资源：当前 GPU 空闲，44.99 GiB 可用总量；数据盘余量约 3.4 TiB。Stage 0 的 2048 合成样本峰值 22.69 GiB 只作容量锚点，真实吞吐未知，formal 条件性工作估计约 10 小时，待数据 token 总量与 pilot 重算。优先风险是考试同源污染导致配额不足、多轮模板剥离 reasoning、真实长序列及完整 cursor 恢复。测试集只用于预先固定文本去重，不按答案/难度或模型表现调规则。
