"""A self-contained byte-level tokenizer that needs no download.

The downstream-training proof must run offline (CLAUDE.md offline-first rule), so
we cannot pull a pretrained tokenizer from the Hub. This builds a deterministic
byte-level tokenizer (256 byte tokens + 4 specials) entirely in-process. It is not
meant for real training quality — it exists so a real `transformers`/`trl` trainer
can ingest agentdata's emitted files and we can watch the loss move.

Swap `build_tiny_tokenizer()` for `AutoTokenizer.from_pretrained("<base-model>")`
to train a real model on the very same emitted JSONL.
"""

from __future__ import annotations

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Minimal ChatML-ish template so SFTTrainer can render {"messages":[...]} rows.
_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "<bos>{{ m['role'] }}\n{{ m['content'] }}<eos>\n"
    "{% endfor %}"
)


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A 260-token byte-level tokenizer with a chat template. Fully offline."""
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {tok: i for i, tok in enumerate(_SPECIALS)}
    for ch in sorted(alphabet):
        vocab[ch] = len(vocab)

    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tok.chat_template = _CHAT_TEMPLATE
    return tok


if __name__ == "__main__":
    t = build_tiny_tokenizer()
    print("vocab size:", t.vocab_size)
    print("encode:", t("hello agentdata")["input_ids"][:12])
