# SFT 与 RL 输出格式合同

Stage 0 使用固定 Qwen3-8B tokenizer；实际 Jinja 模板和完整 token/label 快照存入 template run。`enable_thinking=True` 的 generation prefix 只到 `assistant\n`，模型自行生成 think 标签。`False` 会插入空 think；本项目 SFT 与 RL 主路径显式使用 True，避免额外插入或重复标签。

medical-o1 的 `Complex_CoT` 原样进入 reasoning，`Response` 进入最终答案：

```text
<think>
样本自带的 reasoning
</think>

<answer>样本自带的 final answer</answer>
```

Huatuo 没有 reasoning 字段，使用空 think block；不得根据答案编造 CoT。上游多轮对话的每个 assistant target 均须单独建立监督边界，不能把历史模板移除的 reasoning 当成保留。Stage 0 只验证单轮 fixture，多轮数据规范化和真实配额留给 Stage 1。

`data.format.sft_tokens` 明确 `return_dict=False`，检查完整序列以 prompt tokens 为前缀。只监督 assistant target 与终止 token，prompt labels 为 -100；未来 padding 依据 attention/length mask，而非 `token == EOS`。Transformers 5.5.3 默认返回 BatchEncoding，第一次测试已捕获旧 list 假设错误。

CMExam parser 允许完整 think + answer 或 answer-only，answer 只可含 A–E 和显式分隔符。多选排序规范化，重复字母、非法字母、多个 answer、未闭合标签和答案后的额外文字均返回 None。think 中出现额外 answer 标签也拒绝；未知结果不能默认为错误之外的有效选项。自由文本 SFT 的 answer 无须限制为字母，该限制只属于 CMExam parser。

输入中的保留输出标签和 chat control tokens 必须显式拒绝或在数据处理阶段另立可追溯清理记录，不能默默改变 target。长度截断不能留下假闭合答案；Stage 0 capacity sweep 的合成 token batch 只测显存，不被解释为完成 SFT 数据流水线。

rollout 的 prompt 不含 ground truth 或 explanation；选项和回答格式指令属于 prompt。完整 raw text 保留在 artifact 中。格式 reward 和正确性 reward 分开记录。
