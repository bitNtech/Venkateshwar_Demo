"""Turn finetune/flows.json into finetune/train.jsonl + finetune/val.jsonl.

One JSONL record per flow: a whole call as OpenAI chat messages. The system
message is produced by backend.prompt_builder.PromptBuilder and
backend.conversation.render_template - the same two calls conversation.py makes
on every live turn - so a training example is byte-identical in shape to what
the model will see in production. Reimplementing either here would let the two
drift apart silently, which is the failure this file exists to avoid.

Targets are the TAMIL-SCRIPT line, never the romanised one. The romanisation in
the PDF describes what ASR hands us on the way IN; the agent speaks Tamil script
on the way OUT, and training on the romanised line would teach the model to
answer in Tanglish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

FINETUNE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FINETUNE_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.conversation import render_template  # noqa: E402
from backend.prompt_builder import PromptBuilder  # noqa: E402
from backend.tools import TOOL_SCHEMAS, MockHospitalDb, execute_tool  # noqa: E402

DEFAULT_FLOWS = FINETUNE_DIR / "flows.json"
DEFAULT_TRAIN = FINETUNE_DIR / "train.jsonl"
DEFAULT_VAL = FINETUNE_DIR / "val.jsonl"

# Held out for eval. Deliberately not random: 03 is a short clean happy path,
# 08 carries the "never read out a lab value" clinical boundary, 13 is the only
# multi-MRN registration, and 18 is the emergency override. If the adapter
# breaks any of those four, loss on them moves - which is what eval is for.
VAL_FLOWS = frozenset({3, 8, 13, 18})

# Flow 01's lookup turn, split at the ellipsis the agent actually says while the
# system is being queried. The half before it rides along with the tool call,
# the half after it reads the result back - exactly the two assistant messages
# conversation.py's tool loop produces for one caller turn.
TOOL_FLOW = 1
TOOL_TURN_MARKER = "ARV-118342"
TOOL_SPLIT = "... "
TOOL_CALL_ID = "call_lookup_patient_1"


def load_flows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python finetune/extract_flows.py")
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt_builder() -> PromptBuilder:
    # exemplars_path is deliberately not passed. golden/flow_exemplars.json is
    # the prompt-side substitute for the register this fine-tune installs in the
    # weights, and the post-fine-tune runtime switches it off with a blank
    # CONVERSATION_EXEMPLARS_PATH - so training with it in the prompt would
    # train against a prompt that will not be served. It would also put each
    # flow's own exchanges into that flow's system message, teaching the model
    # to copy from context rather than to speak this way.
    builder = PromptBuilder(
        REPO_ROOT / "golden" / "runtime_core.txt", REPO_ROOT / "golden" / "main_prompt.txt"
    )
    builder.load()
    return builder


def system_message(builder: PromptBuilder, flow: dict) -> str:
    # Only facts the hospital holds before the caller speaks. extract_flows.py
    # explains why mrn is blank for all twenty: every MRN in the corpus is
    # discovered during the call, and seeding one would train the model to state
    # an identifier it never looked up.
    metadata = {
        "agent_name": flow["agent_name"],
        "caller_name": "",
        "caller_mobile": "",
        "mrn": flow["known_mrn"],
        "patient_name": flow["known_patient_name"],
        "last_visit": "",
        "campaign": "",
    }
    return render_template(builder.build(flow["intent"]), metadata)


def merge_consecutive(turns: list[dict]) -> list[dict]:
    """Fold the PDF's back-to-back AGENT lines into one assistant message.

    Four flows show the agent speaking twice with no caller line between. Left
    as two assistant messages they would train the model to keep talking after
    its own end-of-turn token, a state that never occurs at inference because a
    caller utterance always separates two agent turns.
    """
    merged: list[dict] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {**merged[-1], "tamil": f"{merged[-1]['tamil']} {turn['tamil']}"}
        else:
            merged.append(dict(turn))
    return merged


def turns_to_messages(turns: list[dict]) -> list[dict]:
    return [
        {"role": "assistant" if turn["role"] == "AGENT" else "user", "content": turn["tamil"]}
        for turn in merge_consecutive(turns)
    ]


def tool_call_messages(flow: dict) -> list[dict]:
    """Rebuild flow 01's lookup turn as the assistant/tool/assistant triple."""
    turns = merge_consecutive(flow["turns"])
    index = next(
        (
            i
            for i, turn in enumerate(turns)
            if turn["role"] == "AGENT" and TOOL_TURN_MARKER in turn["tamil"]
        ),
        None,
    )
    if index is None:
        raise ValueError(f"flow {TOOL_FLOW}: no agent turn containing {TOOL_TURN_MARKER}")

    spoken = turns[index]["tamil"]
    head, separator, tail = spoken.partition(TOOL_SPLIT)
    if not separator or not head or not tail:
        raise ValueError(f"flow {TOOL_FLOW}: lookup turn does not split at {TOOL_SPLIT!r}: {spoken!r}")

    # The mobile the caller just read out, and the only argument the agent is
    # allowed to pass: it came from the caller, not from the model.
    mobile = "".join(ch for ch in turns[index - 1]["tamil"] if ch.isdigit())
    result = execute_tool(MockHospitalDb(), "lookupPatient", {"mobile": mobile})
    if not result.get("found") or result.get("mrn") != TOOL_TURN_MARKER:
        raise ValueError(f"lookupPatient({mobile}) did not resolve to {TOOL_TURN_MARKER}: {result}")

    messages = [message for message in turns_to_messages(turns[:index])]
    messages.append(
        {
            "role": "assistant",
            "content": head + TOOL_SPLIT.strip(),
            "tool_calls": [
                {
                    "id": TOOL_CALL_ID,
                    "type": "function",
                    "function": {
                        "name": "lookupPatient",
                        "arguments": json.dumps({"mobile": mobile}, ensure_ascii=False),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": TOOL_CALL_ID,
            "content": json.dumps(result, ensure_ascii=False),
        }
    )
    messages.append({"role": "assistant", "content": tail})
    messages.extend(turns_to_messages(turns[index + 1 :]))
    return messages


def build_example(builder: PromptBuilder, flow: dict) -> dict:
    body = tool_call_messages(flow) if flow["flow_number"] == TOOL_FLOW else turns_to_messages(flow["turns"])
    example = {
        "flow_number": flow["flow_number"],
        "intent": flow["intent"],
        "messages": [{"role": "system", "content": system_message(builder, flow)}, *body],
    }
    if any("tool_calls" in message for message in body):
        # Only the example that actually calls a tool carries the schemas. The
        # 22 tool definitions cost ~1.7k tokens in the rendered prompt, and
        # attaching them to all twenty would push every sequence past the 4 GB
        # card's budget to teach nothing the other nineteen demonstrate.
        example["tools"] = TOOL_SCHEMAS
    return example


def token_lengths(examples: list[dict], model: str) -> list[int] | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model)
    # Render to text and encode separately: transformers 5 changed
    # apply_chat_template(tokenize=True) to return a BatchEncoding, so len() of
    # its result counts dict keys, not tokens. The rendered string already
    # carries its own special tokens as text.
    return [
        len(
            tokenizer(
                tokenizer.apply_chat_template(
                    example["messages"], tools=example.get("tools"), tokenize=False
                ),
                add_special_tokens=False,
            )["input_ids"]
        )
        for example in examples
    ]


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flows", type=Path, default=DEFAULT_FLOWS)
    parser.add_argument("--train-out", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val-out", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = parser.parse_args()

    builder = build_prompt_builder()
    flows = load_flows(args.flows)
    examples = [build_example(builder, flow) for flow in flows]

    train = [e for e in examples if e["flow_number"] not in VAL_FLOWS]
    val = [e for e in examples if e["flow_number"] in VAL_FLOWS]
    if len(val) != len(VAL_FLOWS):
        raise ValueError(f"expected {len(VAL_FLOWS)} val flows, matched {len(val)}")

    write_jsonl(args.train_out, train)
    write_jsonl(args.val_out, val)
    print(f"wrote {args.train_out} ({len(train)} examples) and {args.val_out} ({len(val)} examples)")

    lengths = token_lengths(examples, args.model)
    if lengths is None:
        print("transformers not installed - skipping token statistics")
        return 0

    print(
        f"tokens per example: min={min(lengths)} median={int(statistics.median(lengths))} "
        f"max={max(lengths)} (tokenizer: {args.model})"
    )
    for example, length in sorted(zip(examples, lengths), key=lambda pair: -pair[1])[:5]:
        print(f"  longest: flow {example['flow_number']:02d} {example['intent']:<28} {length} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
