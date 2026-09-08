# 依赖策略与候选锁定

审计日期：2026-09-08。版本依据来自官方 release、固定源码、PyPI metadata，而非 2025 示例。区分 **源代码固定**、**metadata 可解**、**运行通过** 三层，不能把第一层写成第三层。

## 推荐候选

| 组件 | 候选固定值 | 选择理由 / 尚需验证 |
|---|---|---|
| Python | 3.12 系列（环境构建时锁 patch） | 上游 x86_64 构建使用 3.12；当前 3.12.7 可用于解析探测 |
| verl | v0.9.0，`483b8a009ba3a97563edee3a19887e4862b8094a` | 8 月 14 日 release；原生 GSPO、LoRA adapter load、FSDP2 engine |
| verl-recipe | `e7f889574b8301cc0f0fc1d57c6d67f31ffeb689` | release 的 gitlink，DAPO 为源码参考；不宣称原样即兼容 |
| vLLM | 0.24.0，`ee0da84ab9e04ac7610e28580af62c365e898389` | verl v0.9.0 的 x86_64 CI 使用 vllm024.dev2；最新观察为 0.28.0，未据此升级 |
| torch / triton | 2.11.0 / 3.6.0 | vLLM 0.24.0 PyPI metadata 固定 torch 2.11.0；不能复用 base 的 2.12.0 |
| CUDA build/runtime | 13.0.2 对应工具链 | verl Dockerfile 与 torch metadata；系统 nvcc 12.0 不适用 |
| Transformers | 5.5.3 | verl CI pin；满足 `>=5.5.3,!=5.6.0,<5.11` 与 vLLM 下限 |
| PEFT | 0.18.1 | 已存在且 metadata 支持；加载/保存与 Transformers 5.5.3 尚需集成验证 |
| datasets / accelerate | 4.8.4 / 1.14.0 | 固定现有版本降低漂移；数据通过 JSON/CSV reader，不依赖旧 dataset scripts |
| sentence-transformers | 5.3.0 | embedding 统一接口；语义有效性与安装兼容性是不同问题 |
| flash-attn | 2.8.3 候选 | 来自上游 Dockerfile；必须匹配 torch、CUDA、Python、C++ ABI、sm_89，尚未构建 |
| bitsandbytes | 不纳入主线 | 主线 BF16 LoRA，无需自动引入量化 |
| Ray / torchdata / tensordict 等 | 由 metadata 解算结果冻结 | tensordict 满足 `>=0.8.0,<=0.10.0,!=0.9.0`；安装后再固定完整环境 |

官方依据：[verl release](https://github.com/verl-project/verl/releases/tag/v0.9.0)、[setup.py](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/setup.py)、[CI](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/.github/workflows/vllm.yml)、[Dockerfile](https://github.com/verl-project/verl/blob/483b8a009ba3a97563edee3a19887e4862b8094a/docker/Dockerfile.stable.vllm)、[vLLM metadata](https://pypi.org/pypi/vllm/0.24.0/json)、[torch metadata](https://pypi.org/pypi/torch/2.11.0/json)。

## 不能忽略的版本矛盾

1. 上游 Dockerfile 默认 Transformers=5.3.0，而 release 的 setup.py 要求至少 5.5.3，CI 又显式安装 5.5.3。因此不原样复制 Dockerfile。
2. Dockerfile 最后用 `--no-deps` 把 NCCL 提高到 >=2.29.7；torch 2.11.0 metadata 却要求 `nvidia-nccl-cu13==2.28.9`。普通 resolver 不能同时满足两者。候选普通环境先保留 torch 的 2.28.9，是否需要 suspend/resume 特性须做 single-GPU 切换探测；不要声称已经通过上游同款 NCCL 配置。如果必须 override，单独创建 build decision、记录 override、ELF/runtime 测试及预期 pip-check 差异；或验证匹配 torch 2.12 的新版 vLLM 后整套升级。
3. DAPO 的 REQUIRED_VERL.txt 自带 rolling SHA `bcb638649a50e58494a8ddd92085ad1174f674b8`，与所选 release SHA 不同。使用它的筛选算法与 controller 参考，在本仓库写有明确补丁范围的 adapter；必须通过导入、config compose、实机 refill/resume，才可冻结。
4. 当前代码进入 `verl/workers/engine/fsdp/transformer_impl.py` 路径。旧 `workers/actor/dp_actor.py` 的扩展教程不能直接套用。

## 安装流程设计（本次不执行安装）

`candidate-requirements.in` 固定主依赖与 verl source SHA；**不是训练环境已认证的 lock**。本次仅运行无安装的 metadata 解析，详细结果见 evidence 下的 resolver 记录。FA2 不参与盲目源码构建。

未来步骤：

1. 建 `.venv-train`，禁用 system-site-packages，设置独立缓存；保留 base 原样。
2. 在匹配 CUDA 13.0.2 的容器/隔离工具链构建；若 Docker 不可用，在项目私有 toolchain 路径装 compiler，不替换 `/usr/bin/nvcc`。容器选定后记录 image digest，当前没有虚构 digest。
3. 将候选源、runtime transitive pins、FA2 wheel/build hash、编译命令、`TORCH_CUDA_ARCH_LIST=8.9` 写入 `env/`。无需引入 Megatron、MoE、vllm-omni 或完整上游开发镜像。
4. `uv pip sync env/train.lock` 后 `pip check`、import、BF16、SDPA/FA2 parity、LoRA save/load、vLLM LoRA/sleep、FSDP2 world_size=1、NCCL 单进程 group、两次权重同步、恢复测试。
5. 满足这些 gate 后，才把 dependency status 从 CANDIDATE 改成 RUNTIME_VALIDATED，并固定 wheel SHA256 / CUDA shared-library 版本。最终 lock 在 SFT 与 RL 比较期间不浮动。

SFT 首先采用 Transformers Trainer + PEFT，SDPA 是可用的低复杂度诊断路径；RL 的 remove-padding/fused 路径如要求 FA2，必须独立验证，不能把 SFT 的 SDPA 成功当作 RL 通过。分析环境可仅 CPU，版本另锁；同一 model/tokenizer/parser 的 hash 不变。

## 模型与数据 source pins

完整 ID/revision 见 [upstream-audit.json](evidence/upstream-audit.json)。Qwen 模型和 tokenizer 均固定 `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，不使用不存在的 `Qwen3-8B-Instruct` 代称，不替换成 Qwen3.5/3.8。`trust_remote_code=False`。所有训练、评估、serving 都从同一快照加载。

MedEmbed small/base/large v0.1 当前可见；候选 small 的 revision 已锁，仅表示可定位，不表示适合中文。中文备选 BGE-M3 revision 已记录，仍属 reward decision proposal。

本次metadata结果：固定verl setup.py的18条install_requires经AST读取，与主候选pins合并，uv无安装解析得到242个packages（Ray2.58.0、tensordict0.10.0、torchdata0.11.0、NCCL2.28.9）。最初直接对git source使用--no-build被拒绝，保留失败原因；替代过程没有执行setup.py或编译扩展。输入/输出SHA与边界见[dependency-resolution.json](evidence/dependency-resolution.json)、[解析候选](evidence/candidate-metadata-resolved.txt)。尚未含flash-attn构建与任何runtime测试，不能据此宣布最终依赖锁定成功。

## Stage 0 installation outcome

The candidate stack is now installed in `.venv-train`, with separate `.venv-analysis`; see [environment manifest](../../env/environment_manifest.json), [rebuild instructions](../../env/README.md), and exact lock files. Core imports, pip check, BF16 CUDA, PEFT backward/reload, native verl FSDP2 and native GSPO have actual run evidence. No standard NCCL override was applied. socksio was added following a real SOCKS ImportError. Native flash-attn remains absent; SFT/FSDP2 explicitly use SDPA. vLLM loads bundled FlashAttention 2 and explicitly uses native sampling to avoid FlashInfer JIT with the system CUDA 12 compiler. Runtime PATH includes the venv's ninja.

A successful import or model load does not close sleep/wake adapter identity: Stage 0 records intermittent output/logprob mismatches, and the final report/selected run determines the accepted boundary checks. V1 and V2 are both present in historical evidence; the final run's execution_environment.json fixes the selected runner. Do not infer a solved bug from a package version.
