# 下游训练闭环 / downstream-training proof

agentdata 的产物是"能直接喂给训练器的数据"。这个目录把这句话**真的跑通**：
agentdata 产出 JSONL → 一个标准训练器原样吃进去 → loss 下降。不是格式校验，是真的反向传播。

> The package claims its output is "training-ready". This directory proves it end to
> end: agentdata emits JSONL, a standard trainer consumes that exact file, and the
> loss falls. Not a schema check — real gradient steps.

## 怎么跑 / run it

```bash
pip install -e '.[train]'          # trl + transformers(<5) + torch + datasets
python train/sft_trl.py            # 种子 -> agentdata --emit chat -> 真·trl.SFTTrainer
python train/dpo_trl.py            # 种子 -> agentdata --emit dpo  -> 真·trl.DPOTrainer
python train/dpo_min.py            # 同上，但不依赖 trl（版本错位时的兜底）
```

> `[train]` extra 把 `transformers` 钉在 `<5`：`trl 0.13` 的 `DPOTrainer` 引用了
> transformers 5 已删除的符号。已验证组合：`trl==0.13 + transformers==4.47`。

无需联网、无需 GPU、无需下载任何模型：学生是一个 2 层、从 config 直接初始化的 GPT-2，
分词器是进程内构造的字节级分词器（`tiny_tokenizer.py`，260 词表）。它们只为"让真实训练器
能把我们的数据吃下去并学到东西"服务，不追求语言质量。

## 实测 / measured (12-row medical fixture, CPU, ~1s each)

| loop | trainer | first loss | last loss | |
|------|---------|-----------:|----------:|---|
| SFT  | `trl.SFTTrainer`（真实） | 5.50 | 2.99 | −46% |
| DPO  | `trl.DPOTrainer`（真实，pinned extra） | 0.69 | 0.003 | −99%，rewards/acc → 1.00 |
| DPO  | 标准 DPO 目标（`dpo_min.py` 兜底） | 0.70 | 0.04 | −94%，pref-acc → 1.00 |

## DPO 的两条路 / two DPO paths

装了 `[train]` extra（钉住 `transformers<5`）后，`dpo_trl.py` 跑的是**真实的 `trl.DPOTrainer`**。
若你的环境里 `trl`/`transformers` 版本错位（比如全局是 transformers 5.x，`trl 0.13` 的
`DPOTrainer` 会因引用已删除符号而无法导入），`dpo_min.py` 在**同一个** `{prompt,chosen,rejected}`
文件上跑标准 DPO 目标，不依赖 trl：

```
L = -log σ( β · [ (logπ_chosen - logπ_rejected) - (logπ_ref_chosen - logπ_ref_rejected) ] )
```

参考模型是初始策略的冻结副本。两者吃的是同一份文件，结果一致（loss 趋零、偏好准确率达 1.0）。
`tests/test_train.py` 优先用真实 `DPOTrainer`，导入失败才回落到 `dpo_min`，并断言 `--emit dpo`
的产物字段与 `trl.DPOTrainer` 的 schema 逐字一致。

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
| `dpo_trl.py` | dpo JSONL → 真实 `trl.DPOTrainer`（需 `[train]` extra） |
| `dpo_min.py` | dpo JSONL → 标准 DPO 目标，不依赖 trl（兜底） |
| `tiny_tokenizer.py` | 离线字节级分词器（260 词表 + chat 模板） |
| `_common.py` | 共享：agentdata 产出、from-config GPT-2、wandb 屏蔽 |
| `fixtures/chat_seed.jsonl` | 12 条医疗问答种子 |
