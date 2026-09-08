# 风险登记

2026-09-08。P0阻止受影响正式run冻结；P1必须在对应验收前关闭/有可审计处理；P2限制结论。负责人默认项目实现工程师，研究合同变化由项目owner确认。本次未关闭需要实际模型实验的风险。

| ID / 级别 | 证据或不确定性 | 触发 / 检测 | 缓解与fallback | 关闭证据 / 阶段 |
|---|---|---|---|---|
| R01 P0 | GPU可见44.99GiB，BF16微探测不是8B | real forward/backward OOM或<3GiB余量 | microbatch1、checkpointing、eager、chunk logits；必要时顺序进程；量化仅proposal | 8B memory/sleep smoke与peak记录 / S1,S4 |
| R02 P0 | torch2.11与NCCL override元数据冲突，nvcc12/runtime13 | import/链接/NCCL/sleep失败 | 独立toolchain13；先标准NCCL；如必要整体换匹配runtime栈，不隐式--no-deps | metadata+pipcheck+ABI+switch receipts / S0 |
| R03 P0 | DAPO gitlink与rolling pin不一致、新engine接口 | config.compose/import/type/单卡resource pool失败 | 固定release、minimal reviewed controller adaptation，CPU与GPU集成测试 | 两组同controller parity / S3,S4 |
| R04 P0 | 原生LoRA未加载可能回退base | adapter缺失、hash不一致、A/B无差异 | 同步barrier/强制adapter check；错误轨迹INVALID | 两轮更新后adapter/logits实证 / S2,S4,S6 |
| R05 P0 | SFT/o1与考试题可能同源 | cross-source exact/near overlap | SFT前全CMExam+CMB question exclusion，heldout优先；不删test改善分数 | dedup cluster和split hash报告 / S1 |
| R06 P0 | MedEmbed英文训练、512 encoder context | 中文UNK/低方差/否定与错解释高sem | train-only64pair诊断，CPU chunk；BGE-M3候选；不用test调weight | encoder comparison+reward decision / S2 |
| R07 P0 | thinking512可能截断最终answer | truncation>5%或closure<95% diagnostic | 两组共享1024/2048 proposal，重新算成本；不改G/5k门槛 | length/protocol freeze / S2,S4 |
| R08 P1 | all-correct hybrid std可非零 | correctness std0但adv std>0 | 同时报3种std；解释intervention删除语义排序信号 | raw group/advantage case / S4,S5 |
| R09 P0 | dynamic acceptance低/相关性高 | 32 refill batches不足8 mixed；a<0.1持续3窗口 | 保存partial与starvation，暂停诊断；不永久黑名单、不降低接受要求 | real256 refill+pilot稳定记录 / S3,S4 |
| R10 P0 | old/model/sampler logprob定义可能混淆 | 初始ratio异常、KL漂移、padding错 | 无top-p/k截断，matched temperature/mask、FP32 logprob、no dropout | numeric+gradient parity / S4 |
| R11 P0 | checkpoint看似存在但optimizer不完整 | 缺marker/hash或step/reset | 原子写、param mapping、两点保留、kill/reload测试 | interrupted-resume receipts / S1,S4 |
| R12 P1 | 崩溃后cost丢失/重复计budget | WAL计数不守恒、unknown尾部 | cost与effective lineage分离；unknown不补0，核引擎记录，必要时重做formal | ledger completeness+rollback tests / S4,S5 |
| R13 P1 | 数据字段/API变化 | CMExam train无annotation、CMB config旧示例 | source/file pinned，strict schema，均匀train抽样；不补造difficulty | schema/dedup+正式count / S1,S2,S5 |
| R14 P1 | 20k或CMB2000去重后不足 | quota无法满足 | 继续同源清洁抽取；不足BLOCKED并proposal，禁止复制/改门槛 | sample IDs/manifests / S1,S5 |
| R15 P1 | reward语义/格式套利，正答案伪推理 | top-sem wrong、重复废话、format-only高频 | 严parser、binary gate、全rawcase；reward修改须两组重开formal | controlled反例与reward bounds / S2,S4 |
| R16 P1 | 磁盘满/RAM压力/数据仅单机 | disk<reserve、RAM avail<16GiB、备份失败 | 工作额度/双写预留、限定Ray和embedding线程、hash备份 | capacity preflight+restore演练 / 全阶段 |
| R17 P1 | 时间可能从数十到数百小时 | pilot校准远超约128h工作假设 | supervisor跨会话、更新资源计划、保留预算；不可几十step结束 | calibrated预算与持久状态 / 全阶段 |
| R18 P1 | 使用test挑checkpoint/调难度 | manifest选点晚于test读取、配置漂移 | 先validation选择与hash冻结，再test；分离data access与评估进程 | comparison/selection ledger / S5,S6 |
| R19 P1 | API content/reasoning拆分或SSE chunk误计token | offline错配、TPOT分母错 | 保留raw reasoning，统一template，token/usage核对、明确chunk估计 | 100consistency+stream rows / S6 |
| R20 P2 | 单训练seed和同源考试限制外推 | 微小delta被夸大、病例cherry-pick | paired CI、cluster sensitivity、正反case、训练seed局限 | Stage5报告/claims ledger |
| R22 P2 | whole-batch单epoch可能令GSPO clip不活跃 | pilot ratio≈1/clipfrac0 | 保留sampling对照，报告限制；优化量改变另立decision | ratio记录与D-008 / S4,S5 |
| R21 P1 | CMExam README使用限制与仓库license含义不同 | 对外分发原始考试题或商业化 | 项目按研究用途；保留来源声明，Git仅结构/允许摘录；商业范围另核 | dataset provenance / S1,S5 |

所有错误、failed/invalid/negative run留manifest/日志。风险未触发不是PASS；例如没有启动服务不等于“未观察到serving mismatch所以验收通过”。
