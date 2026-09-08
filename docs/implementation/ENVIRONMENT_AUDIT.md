# 运行环境审计 — 2026-09-08

本次是 Repository Takeover / Architecture & Execution Planning。没有启动任何 Stage 正式任务，没有下载完整数据集或模型权重，没有测量 8B 吞吐率。硬件与软件均来自本机命令，原始记录见 [environment-20260908-v2.json](evidence/environment-20260908-v2.json)。

## 实际硬件

| 项目 | 实测 / 系统报告 |
|---|---|
| GPU | **1 × NVIDIA RTX 5880 Ada Generation**，不是 RTX A6000 / RTX 6000 Ada / A100 |
| VRAM | nvidia-smi 报告 46,068 MiB = 44.99 GiB；规划按这个可见容量计算 |
| Compute capability | 8.9（sm_89） |
| Driver | 580.173.02；nvidia-smi 显示的 CUDA 13.0 是驱动支持信息 |
| 空闲状态 | 首次检查 GPU 利用率 0%，15 MiB 已用，Xorg 占用少量显存；不是未来独占承诺 |
| 功率上限 | nvidia-smi 报告 285 W |
| CPU | Intel Xeon Platinum 8368Q @ 2.60 GHz，1 socket、38 cores、76 logical CPUs |
| RAM | 117,715,283,968 bytes = 109.63 GiB；首次可用约 99.32 GiB |
| Swap | 8 GiB；不能把 swap 当作稳定训练内存 |
| 系统 | Ubuntu 24.04.3 LTS，x86_64，kernel 7.0.0-28-generic |
| `/data` | `/dev/sda1`，ext4，5.5 TiB 量级，总体可用约 3.4 TiB |
| `/`、`/tmp`、用户目录 | `/dev/nvme0n1p2`，ext4，可用约 1.1 TiB |

空间值是时点快照。设备类型不能由路径推导其实际吞吐；本次未做磁盘性能基准。`/data` 是可跨进程/SSH 会话保留的本地挂载，**尚无第二份备份或机器损坏后的恢复保证**。

## 软件（当前 conda base，而非项目已验证环境）

| 组件 | 当前值 |
|---|---|
| Python / pip / conda / uv | 3.12.7 / 24.2 / 24.11.3 / 0.10.7 |
| Python 路径 | `/home/ubuntu/anaconda3/bin/python` |
| PyTorch | 2.12.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| CUDA toolkit / nvcc | `/usr/bin/nvcc`，12.0.140 |
| cuDNN / NCCL | torch 查询 92000 / 2.29.7 |
| Transformers / PEFT | 5.9.0 / 0.18.1 |
| datasets / accelerate | 4.8.4 / 1.14.0 |
| triton / sentence-transformers | 3.7.0 / 5.3.0 |
| safetensors / huggingface-hub | 0.7.0 / 1.14.0 |
| vLLM / verl / Ray | 未安装于当前解释器 |
| flash-attn / bitsandbytes | 未安装于当前解释器 |
| jsonschema | 4.23.0 |

已列出 conda 环境，存在其他项目的环境；不把它们当成本项目可复用的锁定环境，也不修改它们。CUDA 13 runtime 可执行，不代表 CUDA 12 编译器能构建与其兼容的扩展。新环境需独立 pin，见 [依赖策略](DEPENDENCY_STRATEGY.md)。

## 本次轻量验证

可重复命令（输出文件必须不存在，防止覆盖证据）：

```bash
python scripts/probe_environment.py --output /tmp/new-environment-probe.json --cuda-smoke
```

已留存 run `s0_environment_20260908T100202Z_a4db66f0`，class=DIAGNOSTIC、purpose=environment_probe、stage=0。仅用 seed=0 的 64×64 BF16 矩阵执行前向/反向：标量结果 64、梯度有限、CUDA 可用、BF16 支持。脚本 SHA、实际耗时和峰值分配在 JSON 中。这个微小张量验证**不支持任何 SFT/GSPO 速度或显存结论**。此前一次交互式同形状检查也是环境探测，未作为 Stage 证据。

网络审计只取官方 metadata、源码、模型配置；四个数据文件各读取至多 65,536 bytes，HTTP 206，见 [字段探测记录](evidence/dataset-schema-probe.json)。test 文件只检查 header，不做答案/难度分析，不用于设计超参；临时前缀不进入训练或 Git。没有运行 1k profiling。

## 路径与长任务能力

- 仓库：`/data/WSH/post-train`；本次起始提交 `8af94b2`，初始 main 工作区干净。
- 拟定 bulk root：`/data/WSH/medical-post-train-artifacts`，包括 `models/ datasets/ runs/ cache/`；本次不创建训练内容。
- 拟定独立环境：仓库 `.venv-train/`、`.venv-analysis/`；安装前重新检查空间。
- 临时源码审计：`/tmp/medical-takeover-upstream`；不作为后续唯一证据，关键 URL/revision/hash 已入 Git。
- `systemctl is-system-running` 为 running，`loginctl show-user ubuntu -p Linger` 为 yes；有 tmux。后续使用持久 systemd user unit，重启后先恢复审计，再启动训练；不依赖终端存活。
- 不扫描/输出环境变量全集、访问凭据或其他项目数据；环境记录采用字段白名单。

## 未验证边界

8B 权重加载、真实 LoRA backward 峰值、FSDP2 world_size=1、vLLM sleep/wake、adapter 同步、FA2 编译/内核、真实 checkpoint 恢复都尚未验证。六个 Stage 保持 NOT_STARTED。正式计算预算是基于实机容量的情景估算，不是实测 benchmark。

审计修正：首版probe把`df -BT`解析成TiB block单位，舍入过粗。已改为`df -B1 -T`并重复同样的微型environment probe，v2提供精确bytes与/dev/shm容量；首版JSON与脚本快照保留于evidence，未覆盖。两次都不属于任何Stage执行。Docker命令路径已发现`/usr/bin/docker`，daemon可用性与镜像digest仍未验证。
