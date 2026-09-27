# finetune/ — QLoRA fine-tune of Qwen3-4B on the 20 golden Tamil call flows

Teaches the model the one thing prompting keeps failing at on a small base:
spoken Chennai Tamil in Tamil script, one question per turn, question last,
no invented IDs. Everything here targets **an RTX 2050 with 4 GB of VRAM over
SSH**, so 4-bit QLoRA is the default configuration, not an option.

Training examples use the **condensed runtime prompt** that
`backend/prompt_builder.py` actually ships (`golden/runtime_core.txt` + ONE
section-8 playbook), not the 15k-token master spec. That is what makes 4 GB
feasible at all.

---

## 0. One-time setup on the SSH box

**Use a separate venv.** `../requirements.txt` pins `huggingface_hub==0.23.2`
for the NeMo ASR fork; the training stack needs a far newer hub. Installing
both into one environment breaks ASR. See the header of `requirements.txt`.

```bash
ssh you@the-box
cd /path/to/AICA

python -m venv .venv-finetune
source .venv-finetune/bin/activate

# torch first, from the CUDA index — otherwise pip resolves the CPU wheel
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r finetune/requirements.txt

# optional but recommended on 4 GB; installs last and re-pins to its own set
pip install "unsloth==2025.8.9" "unsloth_zoo==2025.8.8"

python -m bitsandbytes          # self-check: must find a CUDA runtime
```

Run the long steps under `tmux` / `screen` — an SSH drop kills the job
otherwise.

## 1. Extract the transcripts from the PDF

```bash
python finetune/extract_flows.py            # -> finetune/flows.json
```

Reads `golden/tamil_english_call_flows.pdf`. It does **not** read
`golden/flows/*.txt` — those are corrupted (every Tamil conjunct glyph was
dropped by a plain text extractor) and must never be used as a training source.

The script asserts the three edge cases that a careless parser destroys: flow 18
has no agent self-introduction, flow 13 issues two MRNs, flow 20 keeps the
caller turn that begins with `...`. If any assertion fires, the PDF or the
parser changed — fix that before going further.

## 2. Build the dataset

```bash
python finetune/build_dataset.py            # -> finetune/train.jsonl + val.jsonl
```

Each record is one whole call in OpenAI chat-messages form. The system message
is produced by importing `backend.prompt_builder.PromptBuilder` and
`backend.conversation.render_template` — the same two calls the live backend
makes — so training and serving cannot drift apart. Assistant targets are the
**Tamil-script** line; the romanised line is ASR-side input shape and is never
a target.

Split: flows 3, 8, 13, 18 held out for eval, the other 16 for training.

It prints token statistics. Expect roughly:

```
tokens per example: min=3199 median=3803 max=5338
```

Flow 01 is the long one: it is the single tool-call example, so it carries the
22 tool schemas as well.

## 3. Probe the 4 GB fit BEFORE committing to a run

```bash
python finetune/train_lora.py --probe
```

One forward+backward on the **longest** example, then peak VRAM vs total, then
exit. Do this first every time you change `--max-seq-len`, the model, or the
backend (unsloth vs plain transformers). A run that OOMs at step 3 of 4 has
wasted the whole session.

## 4. Train

```bash
python finetune/train_lora.py --output-dir finetune/aruvi-lora
```

Rank 8, alpha 16, dropout 0.1, lr 2e-4 cosine, 2 epochs max, batch 1 with
accumulation 8, gradient checkpointing on, eval + save every 4 steps,
`load_best_model_at_end`, early stopping with patience 1.

On 16 examples **overfitting is the expected outcome, not a risk** — the run is
meant to stop at the first rise in `eval_loss`, and usually will. If `eval_loss`
never rises, the adapter probably has not learned anything either; check the
`supervised (assistant-only) tokens` line, which should read around 29%.

Useful flags: `--model`, `--max-seq-len`, `--epochs`, `--lr`, `--grad-accum`.

## 5. Merge, convert, quantise, register

```bash
python finetune/merge_and_export.py         # -> finetune/aruvi-merged/ (HF format)
```

This merges on **CPU in bf16** (needs ~8 GB RAM, not VRAM) and then prints the
exact llama.cpp and Ollama commands to run next. It deliberately does not run
llama.cpp itself. The commands are, with paths filled in for you:

```bash
python ~/llama.cpp/convert_hf_to_gguf.py finetune/aruvi-merged \
  --outfile finetune/aruvi-merged-f16.gguf --outtype f16

~/llama.cpp/build/bin/llama-quantize \
  finetune/aruvi-merged-f16.gguf finetune/aruvi-merged.gguf Q4_K_M

cd finetune && ollama create aruvi-ft -f Modelfile
```

Then delete the ~8 GB f16 GGUF — nothing reads it again.

## 6. Point the backend at the fine-tuned model

Edit `.env` in the repo root:

```ini
LLM_MODEL=aruvi-ft
```

`LLM_BASE_URL` stays as it is (`http://127.0.0.1:11434/v1`). Restart the
backend. `finetune/Modelfile` sets `num_ctx 8192`, which comfortably holds the
condensed runtime prompt plus a full call's history; the old 32768 allocated a
KV cache far too large for a 4 GB card.

To A/B against the un-finetuned model, register the base one too from the repo
root `Modelfile` (`ollama create aruvi-base -f Modelfile`) and flip `LLM_MODEL`
between `aruvi-base` and `aruvi-ft`.

---

## If it OOMs on 4 GB

Peak memory in a full-sequence backward pass is dominated by the logits, not
the weights: 151669 vocabulary entries × sequence length, materialised and then
upcast for the loss. That makes **sequence length the strongest lever**, in
this order:

1. **Install unsloth.** Its chunked cross-entropy never materialises the full
   logit tensor. This is usually the whole fix, and costs nothing else.
2. **Lower `--max-seq-len`.** 5120 still clears every example but flow 01;
   4096 will make the four longest flows *raise* rather than truncate — that is
   deliberate, since truncating cuts the closing turn, its reference ID and the
   `hangUp` off the end of a call. If you go this way, accept losing those
   examples knowingly rather than silently.
3. **Raise `--grad-accum`.** It does not reduce peak memory (batch size is
   already 1 — the peak is one sequence), but it stabilises the gradient on a
   tiny corpus, so it is the right knob if you have had to drop examples.
4. **Rank 8 is already the floor.** Do not lower it looking for memory; LoRA
   parameters are a rounding error next to activations and logits. Lowering it
   costs capacity and buys nothing.

Also worth checking: nothing else on the box is holding VRAM (`nvidia-smi`) —
the ASR model in particular, if the backend is running on the same machine.
Stop the backend before training.
