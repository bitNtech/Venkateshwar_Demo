"""Merge the trained LoRA adapter into the base weights and save a plain HF model.

Merging happens in bf16/fp16 on CPU, not in 4-bit on the GPU. Two reasons: a
4-bit tensor has no meaningful "add the delta" operation, and a 4B model in
bf16 is ~8 GB of weights - twice what the 2050 has. CPU RAM is the cheap
resource here and this runs once.

There is a known and accepted approximation in doing this: the adapter was
trained against NF4-quantised base weights and is being merged into the full
precision ones, so the merged model is not bit-identical to what was trained.
For a rank-8 adapter on a register/style task the drift is not observable; it
would matter for a high-rank adapter carrying new knowledge.

llama.cpp is NOT invoked from here - it is a separate C++ toolchain with its
own build, and shelling into it from Python hides its failures. The exact
commands are printed instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

FINETUNE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=FINETUNE_DIR / "aruvi-lora")
    parser.add_argument("--out", type=Path, default=FINETUNE_DIR / "aruvi-merged")
    parser.add_argument(
        "--llama-cpp",
        type=Path,
        default=Path("~/llama.cpp").expanduser(),
        help="only used to render the printed next-step commands",
    )
    return parser.parse_args()


def merge(model_name: str, adapter_dir: Path, out_dir: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"no LoRA adapter at {adapter_dir} - run finetune/train_lora.py first")

    print(f"loading base {model_name} in bf16 on CPU (needs ~8 GB RAM, not VRAM)")
    base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    # 2 GB shards keep the write within reach of a small-RAM box and match what
    # llama.cpp's converter expects to walk.
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="2GB")

    # The adapter directory holds the tokenizer train_lora.py saved, which is
    # the one the chat template must come from; fall back to the hub copy only
    # if that directory predates it.
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else model_name
    AutoTokenizer.from_pretrained(str(tokenizer_source)).save_pretrained(str(out_dir))
    print(f"merged model written to {out_dir}")


def print_next_steps(out_dir: Path, llama_cpp: Path) -> None:
    gguf_f16 = out_dir.parent / "aruvi-merged-f16.gguf"
    gguf_q4 = out_dir.parent / "aruvi-merged.gguf"
    modelfile = FINETUNE_DIR / "Modelfile"
    print(
        f"""
Next steps - run these yourself, in order.

1. Convert the merged model to a full-precision GGUF:

     python {llama_cpp}/convert_hf_to_gguf.py {out_dir} \\
       --outfile {gguf_f16} --outtype f16

2. Quantise to Q4_K_M. This is the quantisation the 4 GB card can serve: the
   4B model lands around 2.5 GB, leaving room for the 8192-token KV cache the
   Modelfile asks for.

     {llama_cpp}/build/bin/llama-quantize {gguf_f16} {gguf_q4} Q4_K_M

3. Register it with Ollama. finetune/Modelfile already points FROM at
   {gguf_q4.name}, so run this from the directory holding that file:

     cd {gguf_q4.parent} && ollama create aruvi-ft -f {modelfile}

4. Point the backend at it - edit .env:

     LLM_MODEL=aruvi-ft

5. Delete {gguf_f16.name} once step 2 succeeds; it is ~8 GB and nothing reads
   it again.
"""
    )


def main() -> int:
    args = parse_args()
    merge(args.model, args.adapter, args.out)
    print_next_steps(args.out, args.llama_cpp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
