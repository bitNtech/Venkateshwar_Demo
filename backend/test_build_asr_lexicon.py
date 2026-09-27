"""Self-check for the derivation rules in backend/scripts/build_asr_lexicon.py.

Only the pure functions - no TTS, no ASR, no network. What is guarded here is
the pair of screens that decide WHICH words may enter the generated lexicon,
because that is where the damage would be: an entry that rewrites real Tamil
into the wrong English word is the failure HANDOFF.md Sec6c measured and
rejected, and it would be silent on a live call.
"""

from __future__ import annotations

from .scripts.build_asr_lexicon import (
    MAX_SIMILARITY_TO_REAL_TAMIL,
    derive_real_tamil,
    derive_vocabulary,
    too_close_to_real_tamil,
)

# Two lines in the register the prompt actually teaches, and two in the
# romanised-Tamil convention the flow transcripts also use.
CODE_MIX = [
    "எனக்கு appointment book பண்ணணும்",
    "Cardiology department-க்கு ஒரு slot வேணும்",
]
ROMANISED = [
    "aamaam sir, aduttha vaaram appadi panlaam",
    "aiyo anga alavukku kashtam aagum",
]


def test_the_vocabulary_is_the_code_mix_register_not_every_latin_word() -> None:
    vocabulary = derive_vocabulary(CODE_MIX + ROMANISED)

    # English hospital words, spoken in Latin inside a Tamil sentence.
    for word in ("appointment", "book", "cardiology", "department", "slot"):
        assert word in vocabulary, f"{word} should be in the code-mix vocabulary"

    # Romanised Tamil. Round-tripping these is what would teach the normaliser
    # to rewrite a caller's real Tamil into Latin.
    for word in ("aamaam", "aduttha", "appadi", "panlaam", "alavukku", "aagum", "kashtam"):
        assert word not in vocabulary, f"{word} is romanised Tamil and must be excluded"


def test_a_latin_word_alone_on_a_line_is_not_vocabulary() -> None:
    """Tamil script on the SAME line is the whole discriminator."""
    assert derive_vocabulary(["appointment book panlaam"]) == []
    assert "appointment" in derive_vocabulary(["appointment வேணும்"])


def test_the_similarity_screen_catches_a_near_miss_of_real_tamil() -> None:
    """The exact-collision check cannot see these, and they are the dangerous ones."""
    real = derive_real_tamil("பண்ணா வணக்கம் சொல்லுங்க மருந்து")

    # "அண்ணா" (elder brother) is the measured leak through the code-mix filter.
    assert too_close_to_real_tamil("அண்ணா", real)
    # A genuine transliteration of an English word is nowhere near these.
    assert not too_close_to_real_tamil("அப்பாயின்மென்ட்", real)


def test_an_exact_real_tamil_word_is_always_too_close_to_itself() -> None:
    real = derive_real_tamil("சொல்லுங்க வணக்கம்")
    assert too_close_to_real_tamil("சொல்லுங்க", real)
    assert MAX_SIMILARITY_TO_REAL_TAMIL <= 1.0


def test_real_tamil_is_derived_from_the_text_not_hardcoded() -> None:
    assert derive_real_tamil("appointment வேணும் சார்") == {"வேணும்", "சார்"}
    assert derive_real_tamil("no tamil here") == set()


if __name__ == "__main__":
    test_the_vocabulary_is_the_code_mix_register_not_every_latin_word()
    test_a_latin_word_alone_on_a_line_is_not_vocabulary()
    test_the_similarity_screen_catches_a_near_miss_of_real_tamil()
    test_an_exact_real_tamil_word_is_always_too_close_to_itself()
    test_real_tamil_is_derived_from_the_text_not_hardcoded()
    print("ok")
