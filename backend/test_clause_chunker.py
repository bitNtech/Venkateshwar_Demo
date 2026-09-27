"""Self-check for ClauseChunker's sentence-boundary detection.

Pure logic, no model to stub - exercised the same way
test_conversation.py's render_template tests are (direct calls, no fakes
needed).

A boundary requires an actual trailing space, never just end-of-buffer (see
clause_chunker.py) - so the last clause of any text, even a complete one-shot
reply, only ever comes back via flush(), not feed(). Every test below reflects
that: feed() returns whatever closed *within* the fed text, flush() releases
the rest.
"""

from __future__ import annotations

from .clause_chunker import ClauseChunker


def test_feed_returns_clauses_that_close_within_the_fed_text() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("Cardiology-ல appointment book பண்ணனும். மேலும் கேள்வி இருக்கா?")

    assert clauses == ["Cardiology-ல appointment book பண்ணனும்."]
    assert chunker.flush() == "மேலும் கேள்வி இருக்கா?"


def test_feed_holds_back_incomplete_trailing_text() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("First sentence. Still typing")

    assert clauses == ["First sentence."]
    assert chunker.flush() == "Still typing"


def test_flush_returns_none_once_feed_has_consumed_everything() -> None:
    chunker = ClauseChunker()
    chunker.feed("Complete sentence. ")

    assert chunker.flush() is None


def test_multiple_clauses_in_one_feed_call() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("First sentence. Second sentence. Third?")

    assert clauses == ["First sentence.", "Second sentence."]
    assert chunker.flush() == "Third?"


def test_a_clause_only_closes_once_the_next_chunk_proves_it_with_whitespace() -> None:
    """Mid-stream, buffer-end is not evidence of a real sentence end - only
    whitespace is (a later chunk could still turn "37." into "37.5")."""
    chunker = ClauseChunker()

    assert chunker.feed("Hello") == []
    assert chunker.feed(" world.") == []  # ends right at '.', no proof yet
    assert chunker.feed(" Next") == ["Hello world."]  # now proven by the space
    assert chunker.flush() == "Next"


def test_abbreviation_does_not_trigger_a_false_split() -> None:
    """Without the abbreviation guard, "Dr. " (trailing space and all) would
    incorrectly close as its own clause here."""
    chunker = ClauseChunker()

    clauses = chunker.feed("Dr. Ramanathan is available.")

    assert clauses == []
    assert chunker.flush() == "Dr. Ramanathan is available."


def test_decimal_point_does_not_trigger_a_false_split() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("Room 37.5 is ready. Any questions?")

    assert clauses == ["Room 37.5 is ready."]
    assert chunker.flush() == "Any questions?"


def test_multiple_abbreviations_in_one_turn() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("Rs. 800 fee. Dr. Meera Krishnan, e.g. next Tuesday.")

    assert clauses == ["Rs. 800 fee."]
    assert chunker.flush() == "Dr. Meera Krishnan, e.g. next Tuesday."


if __name__ == "__main__":
    test_feed_returns_clauses_that_close_within_the_fed_text()
    test_feed_holds_back_incomplete_trailing_text()
    test_flush_returns_none_once_feed_has_consumed_everything()
    test_multiple_clauses_in_one_feed_call()
    test_a_clause_only_closes_once_the_next_chunk_proves_it_with_whitespace()
    test_abbreviation_does_not_trigger_a_false_split()
    test_decimal_point_does_not_trigger_a_false_split()
    test_multiple_abbreviations_in_one_turn()
    print("ok")


# --- the opening chunk is the only one whose latency the caller can hear ---

def test_the_first_chunk_closes_on_a_comma_so_audio_starts_sooner() -> None:
    """Measured: an opening sentence of 40 characters produced NO audio for 5.9
    seconds, because nothing could be synthesized until its full stop arrived.
    Every later chunk is generated while the previous one is still playing, so
    only the first one needs this.
    """
    chunker = ClauseChunker()

    clauses = chunker.feed("ICU-க்கு மாலை 5 to 5:30 மட்டும், அதுவும் ஒரு நேரத்துல ஒருத்தர் தான். ")

    assert clauses[0] == "ICU-க்கு மாலை 5 to 5:30 மட்டும்,"


def test_later_chunks_still_wait_for_a_sentence_end() -> None:
    """Breaking every clause on commas would make the whole turn staccato. The
    fast rule applies to the opening only."""
    chunker = ClauseChunker()
    chunker.feed("சரி மேடம், ")  # releases the first chunk

    clauses = chunker.feed("Appointment book பண்ணிட்டேன், reference SMS-ல வரும். ")

    assert clauses == ["Appointment book பண்ணிட்டேன், reference SMS-ல வரும்."]


def test_an_opening_with_no_punctuation_still_releases_at_a_word_boundary() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("இது ரொம்ப நீளமான ஒரு வாக்கியம் எந்த நிறுத்தக்குறியும் இல்லாம வருது ")

    assert clauses, "a long comma-less opening must not hold the audio back"
    assert not clauses[0].endswith(" ")


def test_the_fast_rule_still_respects_abbreviations_and_decimals() -> None:
    chunker = ClauseChunker()

    clauses = chunker.feed("Dr. Ramanathan 37.5 degree fever இருக்கு. ")

    assert clauses == ["Dr. Ramanathan 37.5 degree fever இருக்கு."]


def test_fast_first_chunk_can_be_switched_off() -> None:
    chunker = ClauseChunker(fast_first_chunk=False)

    clauses = chunker.feed("சரி மேடம், அப்படியே பண்ணிடறேன். ")

    assert clauses == ["சரி மேடம், அப்படியே பண்ணிடறேன்."]
