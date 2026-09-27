"""QLoRA fine-tune of Qwen3-4B-Instruct on the 20 golden Tamil call flows.

Target hardware is an RTX 2050 with 4 GB of VRAM over SSH, so 4-bit NF4 with
bitsandbytes is the only configuration that fits - it is the default here, not
an option. Unsloth is used when it imports (its chunked cross-entropy is what
keeps the 151k-token vocabulary's logits off the card); the plain
transformers + peft + bitsandbytes path is a straight fallback with identical
masking, hyperparameters and length checks.

Two safeguards are load-bearing and must not be removed:

  * Assistant-only loss masking. Every span that is not an agent turn is set to
    -100. The system prompt is 2k tokens of rules the model must FOLLOW, not
    reproduce; training on it teaches the model to recite its own instructions.
  * The prefix-stability check. The masking works by rendering messages[:k] and
    treating its length as the boundary of turn k. That is only sound if the
    chat template is prefix-stable, i.e. rendering k messages always produces a
    literal prefix of rendering k+1. Qwen3's template is; a template that
    re-emits a header or closes a block retroactively is not, and would silently
    shift every label by a few tokens. So it is verified, per example, and the
    run aborts rather than training on skewed labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

FINETUNE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Attention + MLP projections. Everything else stays frozen: at 4 GB the
# adapter has to be small, and these are where instruction-following register
# actually lives.
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Measured with the Qwen3 tokenizer on the built dataset: the shortest example
# is ~3.2k tokens and the longest (flow 01, the only one carrying the 22 tool
# schemas) is ~5.3k. The "~1.5k prompt" estimate this project was scoped on
# assumed English; Tamil script costs roughly three tokens per syllable in this
# vocabulary, so runtime_core.txt alone measures 1843. 6144 clears the corpus
# with headroom - see README.md for what to do when it will not fit.
DEFAULT_MAX_SEQ_LEN = 6144

LABEL_IGNORE = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--output-dir", type=Path, default=FINETUNE_DIR / "aruvi-lora")
    parser.add_argument("--train-file", type=Path, default=FINETUNE_DIR / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=FINETUNE_DIR / "val.jsonl")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run one forward+backward on the LONGEST example, report peak VRAM, and exit",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python finetune/build_dataset.py")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render(tokenizer, messages: list[dict], tools, generation_prompt: bool = False) -> str:
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=generation_prompt
    )


def encode(tokenizer, text: str) -> list[int]:
    # The rendered conversation already carries its own <|im_start|> markers;
    # letting the tokenizer add more would shift every span boundary.
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def assert_prefix_stable(tokenizer, messages: list[dict], tools, where: str) -> None:
    full_text = render(tokenizer, messages, tools)
    full_ids = encode(tokenizer, full_text)
    previous = ""
    for count in range(1, len(messages) + 1):
        prefix_text = render(tokenizer, messages[:count], tools)
        if not prefix_text.startswith(previous):
            raise ValueError(
                f"{where}: chat template is NOT prefix-stable at message {count} - rendering "
                f"{count - 1} messages is not a prefix of rendering {count}. Assistant-only "
                "masking cannot be trusted against this template."
            )
        if not full_text.startswith(prefix_text):
            raise ValueError(
                f"{where}: rendering messages[:{count}] is not a prefix of the full render - "
                "the template rewrites earlier turns, so span boundaries are meaningless."
            )
        prefix_ids = encode(tokenizer, prefix_text)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"{where}: token ids for messages[:{count}] are not a prefix of the full "
                "token ids - a turn boundary is being merged into a neighbouring token."
            )
        previous = prefix_text


def build_features(tokenizer, example: dict, max_seq_len: int) -> dict:
    messages = example["messages"]
    tools = example.get("tools")
    where = f"flow {example['flow_number']:02d} ({example['intent']})"

    assert_prefix_stable(tokenizer, messages, tools, where)

    input_ids = encode(tokenizer, render(tokenizer, messages, tools))
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"{where}: {len(input_ids)} tokens exceeds --max-seq-len {max_seq_len}. "
            "Truncating would cut the end of a call - the closing, the reference ID and the "
            "hangUp - so this is a hard stop. Raise --max-seq-len or drop the example."
        )

    labels = [LABEL_IGNORE] * len(input_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        # add_generation_prompt gives the assistant header the template itself
        # would emit at inference time, so the boundary comes from the template
        # rather than from a hand-written "<|im_start|>assistant\n".
        start = len(encode(tokenizer, render(tokenizer, messages[:index], tools, generation_prompt=True)))
        end = len(encode(tokenizer, render(tokenizer, messages[: index + 1], tools)))
        if start >= end:
            raise ValueError(f"{where}: empty supervised span for assistant message {index}")
        labels[start:end] = input_ids[start:end]

    supervised = sum(1 for label in labels if label != LABEL_IGNORE)
    if supervised == 0:
        raise ValueError(f"{where}: nothing supervised - masking is broken")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "_where": where,
        "_supervised": supervised,
    }


def collate(features: list[dict], pad_token_id: int) -> dict:
    width = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for feature in features:
        gap = width - len(feature["input_ids"])
        batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * gap)
        batch["attention_mask"].append(feature["attention_mask"] + [0] * gap)
        batch["labels"].append(feature["labels"] + [LABEL_IGNORE] * gap)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_model(model_name: str, max_seq_len: int):
    """Return (model, tokenizer, backend_name). Unsloth first, transformers second."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        return (*_load_with_transformers(model_name), "transformers+peft+bitsandbytes")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=LORA_TARGETS,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=20250829,
    )
    return model, tokenizer, "unsloth"


def _load_with_transformers(model_name: str):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Double quantisation buys back ~0.4 GB on a 4B model, which on a 4 GB
        # card is the difference between fitting and not.
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    return model, tokenizer


def run_probe(model, features: list[dict], pad_token_id: int) -> int:
    """One forward+backward on the worst-case example, so the fit is measured not hoped."""
    if not torch.cuda.is_available():
        raise SystemExit("--probe measures VRAM and needs a CUDA device")

    longest = max(features, key=lambda f: len(f["input_ids"]))
    batch = collate([longest], pad_token_id)
    batch = {key: value.to(model.device) for key, value in batch.items()}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    loss = model(**batch).loss
    loss.backward()
    model.zero_grad(set_to_none=True)

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    allocated = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    print(f"probe example : {longest['_where']}, {len(longest['input_ids'])} tokens")
    print(f"loss          : {loss.item():.4f}")
    print(f"peak allocated: {allocated:.2f} GiB")
    print(f"peak reserved : {reserved:.2f} GiB   <- this is what has to fit")
    print(f"device total  : {total:.2f} GiB ({torch.cuda.get_device_name(0)})")
    print("VERDICT       :", "fits" if reserved < total * 0.95 else "TOO TIGHT - see README OOM section")
    return 0


def main() -> int:
    args = parse_args()

    model, tokenizer, backend = load_model(args.model, args.max_seq_len)
    print(f"backend: {backend}")

    train_raw = read_jsonl(args.train_file)
    val_raw = read_jsonl(args.val_file)
    train = [build_features(tokenizer, e, args.max_seq_len) for e in train_raw]
    val = [build_features(tokenizer, e, args.max_seq_len) for e in val_raw]

    lengths = [len(f["input_ids"]) for f in train + val]
    supervised = [f["_supervised"] for f in train + val]
    print(
        f"tokens/example min={min(lengths)} median={int(statistics.median(lengths))} "
        f"max={max(lengths)} (limit {args.max_seq_len})"
    )
    print(
        f"supervised (assistant-only) tokens: {sum(supervised)} of {sum(lengths)} "
        f"= {100 * sum(supervised) / sum(lengths):.1f}%"
    )

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if args.probe:
        return run_probe(model, train + val, pad_token_id)

    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

    # Step-based, not epoch-based. 16 training examples at per_device_batch 1 is
    # 16 micro-batches per epoch, which at accumulation 8 is only 2 optimizer
    # steps - epoch-level evaluation would therefore produce two numbers for the
    # whole run and would first look at the model well past the point where a
    # corpus this small has already been memorised. Evaluating on a step cadence
    # is what lets EarlyStopping catch the inflection at all; lowering
    # --grad-accum raises the step count and makes the curve denser still.
    steps_per_epoch = max(1, len(train) // args.grad_accum)
    print(
        f"schedule: {len(train)} examples / accum {args.grad_accum} = {steps_per_epoch} "
        f"optimizer steps per epoch, {int(steps_per_epoch * args.epochs)} total; eval every 4"
    )

    arguments = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        # Paged states keep the optimizer off the card during the spikes a 4 GB
        # device cannot absorb.
        optim="paged_adamw_8bit",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=4,
        save_strategy="steps",
        save_steps=4,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=20250829,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_TorchDataset(train),
        eval_dataset=_TorchDataset(val),
        data_collator=lambda features: collate(features, pad_token_id),
        # Overfitting is the default outcome on 16 examples, not a risk to hedge
        # against: the first eval_loss rise is the signal to stop, immediately.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"adapter written to {args.output_dir}")
    print(f"next: python finetune/merge_and_export.py --adapter {args.output_dir}")
    return 0


class _TorchDataset(torch.utils.data.Dataset):
    def __init__(self, features: list[dict]) -> None:
        # Trainer forwards every key to the model, so the diagnostics used for
        # the printed statistics are dropped here rather than earlier.
        self.features = [
            {k: v for k, v in feature.items() if not k.startswith("_")} for feature in features
        ]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict:
        return self.features[index]


if __name__ == "__main__":
    sys.exit(main())
