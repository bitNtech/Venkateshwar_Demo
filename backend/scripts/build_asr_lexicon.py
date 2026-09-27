"""Derive the ASR's English->Tamil renderings empirically, instead of by hand.

backend/transcript_norm.py ships a hand-written table of the Tamil-script forms
IndicConformer produces for English hospital words. It is correct and it is
small, and "small" is the complaint: a caller who says a word nobody typed into
that table hears it come back as mangled Tamil.

Writing the table by rule does not work, and that is measured - see HANDOFF.md
Sec6c. Tamil script is lossy for English (no b/g/d, unreliable vowels), so
transliterating the ASR's output BACK to English collides: every fuzzy variant
tried corrupted common Tamil words, turning "சொல்லுங்க" (tell me) into
"silence". Turning a caller's real word into the wrong English word is far
worse than leaving one English word transliterated.

This goes the other way round, which is the direction HANDOFF.md named as
untried: generate the expected Tamil form FROM the English word and accept only
exact hits. A form that no English source produced can never be matched, so the
"silence" class of failure is structurally impossible.

It generates the forms by ROUND TRIP rather than by rule - speak the English
word with the same Tamil voice the agent uses, transcribe it with the same
ASR the caller is transcribed by, and record what came back. That is the only
thing that reliably predicts this particular model's output; a grapheme rule
guesses "appointment" as அப்பொஇன்ட்மென்ட் while the model actually emits
அப்பாயின்மென்ட்.

NOTHING here is hardcoded vocabulary:

  vocabulary   every Latin word in golden/ - the prompt, the exemplars and the
               flow transcripts. Add a department to the prompt and it is
               covered on the next build.
  safety net   every Tamil word in golden/. A generated form that collides with
               real Tamil the agent itself writes is REJECTED, which is what
               stops the romanised-Tamil entries in the prompt ("aamaam",
               "anga", "aduttha") from teaching the normaliser to rewrite
               ஆமாம் into Latin.

Output is golden/asr_lexicon.json, written incrementally so an interrupted run
is still usable. transcript_norm.py merges it UNDER the hand table, so a
generated entry can only ever add coverage, never overwrite a measured one.

    .venv/Scripts/python.exe -m backend.scripts.build_asr_lexicon --limit 200
"""

from __future__ import annotations

import argparse
from collections import Counter
import difflib
import json
import logging
import pathlib
import re
import sys
import time

import numpy as np

from ..settings import AudioSettings, TtsSettings

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
GOLDEN = _REPO_ROOT / "golden"
OUTPUT = GOLDEN / "asr_lexicon.json"

_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_TAMIL_WORD_RE = re.compile(r"[஀-௿]+")
_TAMIL_CHAR_RE = re.compile(r"[஀-௿]")

# Below this a token is too short to be safely reversible: three Tamil
# characters is roughly two syllables, and short forms are where collisions
# with real Tamil words come from (the measured "டெஸ்ட்"/"dust" class).
MIN_TAMIL_CHARS = 4
MIN_ENGLISH_CHARS = 4


def _golden_text() -> str:
    parts = []
    for path in sorted(GOLDEN.rglob("*")):
        if path.is_file() and path.suffix in {".txt", ".json"} and path.name != OUTPUT.name:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def derive_vocabulary(lines: list[str], min_occurrences: int = 1) -> list[str]:
    """The English words the agent uses in CODE-MIX, and only those.

    Taking every Latin word in golden/ does not work, and the reason is worth
    stating: the flow transcripts write plenty of Tamil in Latin letters
    ("aamaam", "aduttha", "appadi", "annaikku", "aiyo"). Round-tripping those
    would teach the normaliser to rewrite real Tamil into Latin, which is the
    exact corruption HANDOFF.md Sec6c measured and rejected.

    The discriminator is the register itself. This prompt teaches Tamil-English
    code-mix, so a genuine English hospital word appears as a LATIN ISLAND
    inside a Tamil-script sentence ("எனக்கு appointment book பண்ணணும்"), while
    romanised Tamil appears in lines that are romanised throughout. Requiring
    Tamil script ON THE SAME LINE keeps the first and drops the second.

    Measured on golden/, against 16 hand-picked words of each kind:
        romanised Tamil admitted   1/16   (only "anna", caught by the
                                           similarity screen below)
        English vocabulary kept   15/16   (only "admission" lost)
    """
    counts: Counter[str] = Counter()
    for line in lines:
        if not _TAMIL_CHAR_RE.search(line):
            continue
        for word in _LATIN_RE.findall(line):
            counts[word.lower().strip("-'")] += 1
    return sorted(
        w for w, n in counts.items() if len(w) >= MIN_ENGLISH_CHARS and n >= min_occurrences
    )


# How similar a generated form may be to a word the agent writes in real Tamil.
# The exact-collision check is the first line of defence; this catches the near
# misses it cannot, which are the dangerous ones - "அண்ணா" (elder brother)
# scoring 0.80 against "பண்ணா" is a romanised-Tamil entry that slipped the
# code-mix filter. Deliberately strict: losing an English word costs coverage,
# admitting a Tamil one costs correctness.
MAX_SIMILARITY_TO_REAL_TAMIL = 0.80


def too_close_to_real_tamil(form: str, real_tamil: set[str]) -> bool:
    """Whether `form` is a near-miss of a word the agent writes in Tamil."""
    for word in real_tamil:
        if abs(len(word) - len(form)) > 3:
            continue
        if difflib.SequenceMatcher(None, form, word).ratio() >= MAX_SIMILARITY_TO_REAL_TAMIL:
            return True
    return False


def derive_real_tamil(text: str) -> set[str]:
    """Every Tamil word the agent's own prompt material uses.

    This is the safety net. Any generated form landing in here is a word the
    agent genuinely writes in Tamil, so rewriting it to Latin would corrupt a
    real caller turn - exactly the failure that killed the fuzzy approaches.
    """
    return set(_TAMIL_WORD_RE.findall(text))


def _to_asr_rate(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.int16)
    length = int(len(samples) * target_rate / source_rate)
    resampled = np.interp(
        np.linspace(0, len(samples), length, endpoint=False),
        np.arange(len(samples)),
        samples.astype(np.float32),
    )
    return resampled.astype(np.int16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="stop after N words (0 = all)")
    parser.add_argument("--min-occurrences", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    text = _golden_text()
    vocabulary = derive_vocabulary(text.splitlines(), args.min_occurrences)
    real_tamil = derive_real_tamil(text)
    if args.limit:
        vocabulary = vocabulary[: args.limit]

    print(f"vocabulary derived from golden/: {len(vocabulary)} Latin words")
    print(f"real-Tamil safety net:           {len(real_tamil)} Tamil words")

    from ..asr import IndicConformerAsr
    from ..transcript_norm import _ENGLISH_WORDS
    from ..tts import EdgeTts

    tts = EdgeTts(TtsSettings())
    tts.load()
    audio_settings = AudioSettings()
    asr = IndicConformerAsr(audio_settings)
    asr.load()

    existing: dict[str, str] = {}
    if OUTPUT.exists():
        existing = json.loads(OUTPUT.read_text(encoding="utf-8")).get("lexicon", {})
        print(f"resuming from {len(existing)} already-built entries")

    lexicon: dict[str, str] = dict(existing)
    seen_english = {v for v in lexicon.values()}
    rejected = Counter()
    started = time.perf_counter()

    for index, word in enumerate(vocabulary, 1):
        if word in seen_english:
            continue
        try:
            spoken = tts.synthesize(word, audio_settings.language)
            if not spoken.samples.size:
                rejected["no audio"] += 1
                continue
            samples = _to_asr_rate(spoken.samples, spoken.sample_rate, audio_settings.sample_rate)
            # Deliberately the RAW model output: asr.transcribe() would apply
            # the normaliser we are building, and feeding its own output back
            # in would make the result depend on how much it already knows.
            heard = asr.transcribe_raw(samples, audio_settings.language)
        except Exception as error:  # a degraded TTS link must not kill the run
            rejected[f"error: {type(error).__name__}"] += 1
            continue

        form = heard.strip()
        if not form or not _TAMIL_CHAR_RE.search(form):
            # The ASR already produced Latin - nothing to rewrite.
            rejected["not tamil script"] += 1
        elif len(_TAMIL_CHAR_RE.findall(form)) < MIN_TAMIL_CHARS:
            rejected["too short to be safe"] += 1
        elif " " in form:
            rejected["multi-token"] += 1
        elif form in real_tamil:
            # THE important one: this is a word the agent writes in real Tamil.
            rejected["collides with real tamil"] += 1
        elif too_close_to_real_tamil(form, real_tamil):
            rejected["near-miss of real tamil"] += 1
        elif form in _ENGLISH_WORDS:
            rejected["already measured by hand"] += 1
        elif form in lexicon and lexicon[form] != word:
            # Two English words produce the same Tamil. Neither is safe.
            del lexicon[form]
            rejected["ambiguous"] += 1
        else:
            lexicon[form] = word
            seen_english.add(word)

        if index % 25 == 0 or index == len(vocabulary):
            rate = index / max(time.perf_counter() - started, 1e-6)
            OUTPUT.write_text(
                json.dumps(
                    {
                        "_comment": "GENERATED by backend/scripts/build_asr_lexicon.py - do not hand-edit.",
                        "lexicon": dict(sorted(lexicon.items())),
                    },
                    ensure_ascii=False,
                    indent=1,
                ),
                encoding="utf-8",
            )
            print(
                f"  {index}/{len(vocabulary)}  kept={len(lexicon)}  "
                f"{rate:.1f} words/s  eta={(len(vocabulary)-index)/max(rate,1e-6)/60:.1f} min"
            )

    print(f"\nkept {len(lexicon)} generated mappings -> {OUTPUT}")
    for reason, count in rejected.most_common():
        print(f"  rejected {count:5}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
