# Stage 0 — Runtime Foundation & Compatibility Gates

2026-09-08。Stage 0 VERIFIED。所有选定兼容性 probe 和全量bulk-hash verifier均PASS；验收收据为 [Stage 0 verifier receipt](../../experiments/stage0/verification-final.json)。Stage 1–6 仍是 NOT_STARTED，正式预算和进度均未变。本轮没有正式 SFT/RL、test-set evaluation 或 serving benchmark。

## 1. 实际机器与环境

单张 RTX 5880 Ada Generation，44.99 GiB 可见显存，driver 580.173.02；Xeon Platinum 8368Q，38 cores / 76 threads，约109.63 GiB RAM。环境证据：[s0_environment_20260908T135907_d1144c](../../experiments/stage0/s0_environment_20260908T135907_d1144c/attempt_001/summary.json)。独立 Python 3.12.7 venv；未修改 conda base。

## 2. 最终依赖与可重建性

torch 2.11.0+cu130 / CUDA 13.0，vLLM 0.24.0，verl 0.9.0 @ 483b8a009ba3a97563edee3a19887e4862b8094a，Transformers 5.5.3，PEFT 0.18.1，Ray 2.58.0, NCCL 2.28.9。native flash-attn 未安装；actor 显式 SDPA、remove_padding=False，vLLM 使用自带 FlashAttention 2。最终 rollout 显式 V1 runner、native sampler、VLLM_BATCH_INVARIANT=1，实际 shrink split_k=1。

完整依赖哈希、verl wheel/source 哈希、安装日志和命令见 [env](../../env/README.md) 与 [environment manifest](../../env/environment_manifest.json)。pip check、关键包 import 和真实 CUDA/FSDP/vLLM 执行都有独立证据；不是只做 metadata resolve。

## 3. 真实 Qwen3-8B snapshot

官方 Qwen/Qwen3-8B @ b968826d9c46dd6066d109eabc6255188de91218；15 个官方文件，16,397,461,266 bytes，五个 safetensors 分片全部按官方 LFS SHA-256 校验，tokenizer/config 也有文件级 hash。固定路径文件设为只读。[s0_snapshot_20260908T132515_ec08e3](../../experiments/stage0/s0_snapshot_20260908T132515_ec08e3/attempt_001/summary.json)。Xet/HTTP 中断和范围恢复并发保护退出均保留，未替换为小模型或量化权重。

## 4–6. BF16 LoRA capacity、显存和吞吐

真实 8B base BF16，r32/alpha64，七类 projections，87,293,952 个可训练参数，microbatch=1，gradient checkpointing，use_cache=False。三个长度各一次 optimizer update，另一次受控 continuation；全部 loss/grad 有限且 adapter 参数变化。[s0_lora_20260908T132530_c22c36](../../experiments/stage0/s0_lora_20260908T132530_c22c36/attempt_001/summary.json)。

| seq | NVML peak GiB | torch allocated GiB | torch reserved GiB | forward/backward s | update-phase tokens/s | loss | grad norm | delta L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 18.562 | 16.956 | 17.537 | 0.3195 / 0.3877 | 584.4 | 0.0003131 | 0.007536 | 0.5821 |
| 1024 | 20.031 | 18.295 | 19.172 | 0.1898 / 0.4505 | 1541.4 | 2.68e-06 | 9.023e-05 | 0.3484 |
| 2048 | 22.691 | 20.333 | 21.811 | 0.3927 / 0.9524 | 1499.0 | 4.563e-08 | 2.931e-06 | 0.2416 |

最大**已测试**长度为2048，不是硬件最大容量结论；该点剩余显存 22.297 GiB。数据是重复的合成 capacity tokens，后半序列有监督，不能用于医学质量判断。吞吐分母为 forward+backward+optimizer，不含取数、测量时的 CPU 参数复制、checkpoint、加载和验证。低 loss 主要反映重复输入的容易程度，不是 SFT 收敛。正式有效吞吐仍需 Stage 1 合同 smoke/pilot 测量。

## 7–9. vLLM、adapter identity 和单卡切换

[s0_vllm_20260908T134647_8251b8](../../experiments/stage0/s0_vllm_20260908T134647_8251b8/attempt_001/summary.json) 使用 BF16 / TP1 / max_model_len4096 / max_num_seqs16 / gpu_memory_utilization0.65 / eager。冷加载 19.852s；峰值约 29.232 GiB。vLLM 的 torch allocated/reserved=0 是**父 probe 进程**计数，不能解释为 engine 不占显存；真实 engine 总占用以 NVML 为准。

base vs adapter 匹配 prompt-token logprob 最大差异 1.608544；再次 base 调用是负对照。冷/热 adapter 连续调用误差为0。首次 sleep 7.261s、wake 1.469s，睡眠期间运行独立 BF16 LoRA actor backward；其加载/更新/退出约13.604s。随后再验证两次 sleep/wake，三次的 token 和匹配 prompt logprob 都一致。后两次 sleep 约0.84s、wake约1.20s，不能把较快的热状态当作首次切换成本。

actor 产生的新 adapter SHA 与旧 SHA 不同，vLLM 新 LoRARequest 的匹配 logprob 变化0.499998。原 adapter、更新 adapter、raw outputs 和子进程日志均留存。16 条×32-token 的内部小批量计时：base 419.44 output tokens/s，LoRA 114.14；包含该形状的激活/调度开销，不是长期吞吐或 API benchmark。

默认非 batch-invariant 路径曾在 V1/V2 都失败，且对照证明 LoRA 在睡眠**之前**就波动；base 重复调用稳定。采用原生 batch-invariant 选项后通过这些检查。证据支持该缓解方案，不能仅凭本测试断言所有 batch shapes、长序列和625次同步都稳定。精确归因到某一个 kernel 仍是推断；详见 D-013/D-016，原失败未删除。

## 10–11. verl / GSPO / clipping

[s0_verl_20260908T133100_897d2d](../../experiments/stage0/s0_verl_20260908T133100_897d2d/attempt_001/summary.json) 实际安装和 import、Hydra compose、native FSDPEngineWithLMHead FSDP2 单卡初始化、LoRA加载、forward/backward/optimizer_step 全部执行；128-token loss/grad有限，峰值约20.639 GiB。原生 NaiveRewardManager 接收并保留 score/acc/sem/format 字典。sem=0只是接口 fixture，不是医学奖励结果。

[s0_numeric_20260908T131331_3f5986](../../experiments/stage0/s0_numeric_20260908T131331_3f5986/attempt_001/summary.json) 在 CPU 和 GPU 调用 upstream GSPO/GRPO；覆盖正负 advantage、上下 clipping、sequence ratio、非1 ratio、不同 response 长度和恶意 padding 值，并检查有限梯度、padding零梯度。release 内部固定 seq-mean-token-mean；pg_clipfrac_lower 是上游占位0，不能用它推断没有 lower clipping。

[s0_minibatch_20260908T133333_f32dbe](../../experiments/stage0/s0_minibatch_20260908T133333_f32dbe/attempt_001/summary.json) 在真实8B上用本地微批累积调用 native loss，8个合成 prompt × G4，两个条件从同一 adapter 状态开始，共3次更新。8-prompt单mini：ratio=1、clip=0；4-prompt第一mini相同，第二mini ratio范围0.793–1.106、clip=0.875。初始平均loss为0仍可有非零梯度；不能把零loss误读为没有训练信号。D-014仅提出未来pilot的mini=4候选，未修改正式设置。**完整 Ray trainer、rollout correction 和 Stage 3 controller 尚未执行**。

## 12. 中文 semantic reward

[s0_semantic_20260908T134138_110db5](../../experiments/stage0/s0_semantic_20260908T134138_110db5/attempt_001/summary.json)，7类 synthetic pairs，CPU8线程，固定 MedEmbed/BGE-M3 snapshots：

| encoder | synonym | unrelated | negation | number | entity | 8 texts encode s | max tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| abhinand/MedEmbed-small-v0.1 | 0.9626 | 0.7343 | 0.9914 | 0.9691 | 0.9482 | 0.201 | 512 |
| BAAI/bge-m3 | 0.9839 | 0.4276 | 0.9725 | 0.9946 | 0.8812 | 15.291 | 8192 |

重复废话、超长截断分数及实际编码长度见raw metrics。MedEmbed实际截断到512，BGE-M3到8192。两者都把剂量数字替换排在同义改写前；BGE的无关文本分离更好不足以证明其医学奖励可靠。结论 OPEN，未替换 encoder。这个8文本混合延迟包含极长文本，不能直接当作线上每条短解释的服务成本。

## 13. Response length

[s0_length_20260908T133647_20874e](../../experiments/stage0/s0_length_20260908T133647_20874e/attempt_001/summary.json)，只读固定 CMExam train.csv 的前32条，不按标签筛选；每个条件128条完整保留。此时使用同一非 batch-invariant **base** runtime；最终 LoRA runtime 的数值设置不同，该测量不能冒充最终策略评测。

| max_response | closure | truncation | strict format | accuracy | mean tokens | generated tokens | output tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 12.50% | 87.50% | 12.50% | 12.50% | 504.203 | 64538 | 514.6 |
| 1024 | 77.34% | 22.66% | 64.84% | 63.28% | 751.211 | 96155 | 490.8 |

1024明显改善但仍未达到closure>=95%/truncation<=5%。D-015提出两组共同提高候选预算，并要求SFT后train/validation重新诊断；不能直接把1024称为已充分。未测2048输出，不外推其正确率。

## 14. Checkpoint / resume

[s0_reload_20260908T132639_9b0d8f](../../experiments/stage0/s0_reload_20260908T132639_9b0d8f/attempt_001/summary.json)：原进程训练到step3，保存adapter、optimizer、scheduler、Python/NumPy/CPU/CUDA RNG和trainable digest，原进程另做step4参考后退出；新进程恢复step3继续step4。重载logits最大误差 0.0，继续训练参数最大误差 0.0，scheduler一致、optimizer states恢复、global_step=4。不是仅PEFT warm-start。完整真实dataloader cursor和formal FSDP distributed checkpoint恢复仍留给各自阶段。

## 15. Bad/system cases 与证据保留

实际案例包括 SOCKS依赖缺失、ninja PATH、tokenizer返回类型、Hydra/ActorConfig必填项、下载中断与并发保护、默认LoRA数值波动、512截断、语义数字/否定反例、ratio=1和非零clipping。未观察到8B OOM、adapter文件重载误差或resume误差；没有伪造这些失败。案例索引见 [case coverage](../../experiments/stage0/case_coverage.json)，完整 [run inventory](../../experiments/stage0/run_inventory.json) 包含失败和被后续诊断替代的运行。

初始bootstrap在MVP之前，部分下载进度只有tool transcript；精确开始时间未采集，保留null。早期config/template/环境检查保存source hashes，后续真实8B运行还保存source.zip。stdout/stderr的最终hash另行seal。bulk artifacts只在本机数据盘；manifest不是外部备份。

## 16–17. 风险和 Stage 1 readiness

R01容量在2048范围内缓解，R02依赖/ABI在当前路径内缓解，R04在显式batch-invariant和边界断言下缓解，R11受控LoRA恢复缓解；没有把它们泛化为整个正式训练已验收。R06语义、R07长度、R10跨引擎温度/logprob parity、数据去重、自然mixed acceptance、完整Ray循环、长任务重启及外部备份仍OPEN。详见 [Risk Register](../implementation/RISK_REGISTER.md)。

运行环境可进入Stage 1数据准备和合同smoke；Stage 1–6状态仍NOT_STARTED。sft --mode smoke目前是Stage 0容量测试入口；medical_o1/Huatuo规范化和mask基础已存在，正式Stage 1真实配额、heldout去重、dataloader与合同smoke须另行实现/执行。本轮禁止Stage1 FULL仍有效。

## 18. GPU-hours重新校准

[Calibration](../../experiments/stage0/compute_calibration.json) 按实测SFT update phase、LoRA小批量吞吐、base输出长度和切换记录重算，显式区分Measured/Estimated/Unknown。条件性主线估计：optimistic 167.03h / working 339.95h / adverse 1545.04h，均不是确定性工期。工作情景假设a=0.35、共享1024候选、平均751输出tokens、625updates；采用mini=4会改变更新开销，尚未计入。正式LoRA decode和post-SFT长度仍Unknown；a趋0时无有限上界。

## 19. 30秒面试故事

我先用真实8B证明单卡BF16 LoRA到2048能反传，再做跨进程完整恢复和base/adapter对照。第一次vLLM切换看似成功，logprob检查却发现LoRA默认数值不稳定。对照排除了只在sleep发生的假设，启用原生batch-invariant后连续三次切换一致，并确认新adapter实际生效。与此同时，实验发现512输出预算不足、单mini的GSPO clipping不活跃，促成了有原始证据的参数proposal，而没有提前启动正式训练。

## 20. 可深入追问

- frozen 8B为何仍需反传激活，LoRA参数/Adam状态为何远小于base？
- 为什么greedy文本相同不足以证明adapter加载？为什么一次重复成功也不能关闭风险？
- 原子归约顺序、BF16舍入与极窄GSPO clipping阈值如何相互影响？
- one-mini/one-epoch为何ratio=1，但loss为0仍可能有梯度？
- score字典为何必须保留独立acc，而不能让scalar reward冒充正确性？
- PEFT warm-start与恢复optimizer/scheduler/RNG/step的区别是什么？
- 为什么base batch throughput、LoRA throughput和正式GPU占用小时不能互换？
- 为什么语义相似度高不能证明剂量或否定语义正确？
- 如何用unique prompt exposure检测Dynamic Sampling是否反复使用少量boundary题？
