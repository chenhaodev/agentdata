# 下游训练闭环 / downstream-training proof

agentdata 的产物是"能直接喂给训练器的数据"。这个目录把这句话**真的跑通**：
agentdata 产出 JSONL → 一个标准训练器原样吃进去 → loss 下降。不是格式校验，是真的反向传播。

> The package claims its output is "training-ready". This directory proves it end to
> end: agentdata emits JSONL, a standard trainer consumes that exact file, and the
> loss falls. Not a schema check — real gradient steps.

## 怎么跑 / run it

```bash
pip install -e '.[train]'          # trl + transformers + torch + datasets
python train/sft_trl.py            # 种子 -> agentdata --emit chat -> 真·trl.SFTTrainer
python train/dpo_min.py            # 种子 -> agentdata --emit dpo  -> 标准 DPO 目标
```

无需联网、无需 GPU、无需下载任何模型：学生是一个 2 层、从 config 直接初始化的 GPT-2，
分词器是进程内构造的字节级分词器（`tiny_tokenizer.py`，260 词表）。它们只为"让真实训练器
能把我们的数据吃下去并学到东西"服务，不追求语言质量。

## 实测 / measured (12-row medical fixture, CPU, ~1s each)

| loop | trainer | first loss | last loss | |
|------|---------|-----------:|----------:|---|
| SFT  | `trl.SFTTrainer`（真实） | 5.50 | 2.99 | −46% |
| DPO  | 标准 DPO 目标（自带实现） | 0.70 | 0.04 | −94%，pref-acc → 1.00 |

## 为什么 DPO 是自带实现 / why DPO is a built-in objective

`trl.SFTTrainer` 在本机正常工作，所以 SFT 用的是**真实的 TRL 训练器**。
但本环境里 `trl==0.13` 与 `transformers==5.9` 版本错位，`trl.DPOTrainer` 无法导入
（它引用了 transformers 5 已删除的符号）——这是环境的版本问题，不是数据的问题。
为了不污染用户的全局环境去降级依赖，`dpo_min.py` 直接在**同一个** `{prompt,chosen,rejected}`
文件上实现标准 DPO 目标：

```
L = -log σ( β · [ (logπ_chosen - logπ_rejected) - (logπ_ref_chosen - logπ_ref_rejected) ] )
```

参考模型是初始策略的冻结副本。把 `tiny_gpt2()` 换成 `from_pretrained(...)` 就是一次
（最小但标准的）真实 DPO 微调。`tests/test_train.py` 另有断言：`--emit dpo` 的产物
字段与 `trl.DPOTrainer` 的 schema 逐字一致。

## 换成真实模型 / scale to a real model

两处替换即可，数据侧一行都不用改：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tok   = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
```

然后把 `--seed` 指向任意 agentdata 产出：

```bash
agentdata build --source local:your_corpus.jsonl --emit chat --size 5000
python train/sft_trl.py --seed out/dataset.chat.jsonl --steps 500
```

## 文件 / files

| file | what |
|------|------|
| `sft_trl.py` | chat JSONL → 真实 `trl.SFTTrainer`，断言 loss 下降 |
| `dpo_min.py` | dpo JSONL → 标准 DPO 目标，断言 loss 下降 |
| `tiny_tokenizer.py` | 离线字节级分词器（260 词表 + chat 模板） |
| `_common.py` | 共享：agentdata 产出、from-config GPT-2、wandb 屏蔽 |
| `fixtures/chat_seed.jsonl` | 12 条医疗问答种子 |
