"""Extract the 20 golden call transcripts from golden/tamil_english_call_flows.pdf.

golden/flows/*.txt are NOT a usable source. They were produced by a plain text
extractor, and the PDF's Tamil is set in a subset-embedded NotoSansTamil whose
/ToUnicode covers only the glyphs the shaper did not touch. Every conjunct the
shaper produced - consonant+virama, consonant+vowel-sign - has no reverse
mapping, so a plain extractor silently DROPS it: "வணக்கம்" comes out "வணக".
Training on that would teach the model to spell Tamil wrong.

pdfplumber keeps those glyphs as "(cid:NN)" placeholders instead of dropping
them, which makes the text recoverable: the subset has only 57 unmapped glyph
ids across the whole document, and each one is a fixed Tamil cluster. The font
program itself carries no cmap, no post and no GSUB table (it is a subset), so
the mapping cannot be derived at runtime - it is transcribed below once, and
every reconstructed line is then structurally validated (no orphan vowel sign,
no orphan virama) so a wrong or missing entry fails the run instead of quietly
producing malformed Tamil.

Output: finetune/flows.json, consumed by finetune/build_dataset.py.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
import sys

PDFPLUMBER_MISSING = """\
No usable PDF library found.

finetune/extract_flows.py needs pdfplumber specifically. pypdf alone cannot be
used as a fallback: it drops the unmapped Tamil conjunct glyphs entirely, which
is exactly the corruption already visible in golden/flows/*.txt.

    pip install pdfplumber==0.11.10

"""

try:
    import pdfplumber
except ImportError:  # pragma: no cover - environment guard, not logic
    sys.stderr.write(PDFPLUMBER_MISSING)
    raise SystemExit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = REPO_ROOT / "golden" / "tamil_english_call_flows.pdf"
DEFAULT_OUT = Path(__file__).resolve().parent / "flows.json"

# The 57 glyph ids the subset font leaves unmapped, transcribed from their
# rendered form and cross-checked against the romanised line printed under each
# Tamil line in the PDF (e.g. cid 108 sits where "thookkam" is romanised, so it
# is தூ). Several vowel signs appear more than once because the font ships
# contextual alternates of the same sign - 26/65/69 are all plain ி.
CID_TO_TAMIL: dict[int, str] = {
    21: "க்", 23: "ம்", 25: "ரு", 26: "ி", 28: "ல்",
    29: "டி", 30: "ஸ்", 35: "லி", 36: "ட்", 40: "ன்",
    42: "த்", 46: "சு", 49: "ங்", 50: "ளு", 51: "கு",
    53: "ப்", 54: "ண்", 58: "னு", 60: "ர்", 61: "லு",
    63: "மு", 65: "ி", 69: "ி", 72: "து", 74: "ச்",
    75: "கீ", 76: "டு", 77: "ஞ்", 78: "ந்", 79: "மூ",
    80: "பு", 81: "யு", 85: "ணு", 88: "ஷ்", 90: "வு",
    92: "ள்", 93: "கூ", 97: "தீ", 98: "னீ", 99: "ழு",
    101: "றீ", 103: "நீ", 104: "சீ", 105: "ற்", 106: "று",
    107: "வீ", 108: "தூ", 109: "வ்", 110: "ய்", 112: "பீ",
    113: "டீ", 114: "மீ", 116: "ழ்", 118: "பூ", 119: "ஜ்",
    120: "நு", 122: "ஸ்ரீ",
}

_CID_RE = re.compile(r"\(cid:(\d+)\)")

# Tamil ெ ே ை are prebase signs: the renderer emits them to the LEFT of their
# consonant, so extraction yields visual order and they must be moved back
# after it. ொ ோ ௌ are split by the renderer into a prebase half and a postbase
# half around the consonant, so they also have to be recomposed.
_PREBASE_SIGNS = frozenset("ெேை")
_SPLIT_VOWELS = {("ெ", "ா"): "ொ", ("ே", "ா"): "ோ", ("ெ", "ௗ"): "ௌ"}

_TAMIL_CONSONANTS = ("க", "ஹ")  # inclusive range U+0B95..U+0BB9
_VOWEL_SIGNS = frozenset("ாிீுூெேைொோௌ்ௗ")

# Body text, the grey romanised line and the page footer are set in three
# distinct greys at three distinct sizes. Colour is what separates a Tamil turn
# from its romanisation reliably: a turn whose content is only digits ("98407
# 21534.") is set in the Tamil font on BOTH lines, so font name alone mis-sorts
# it, and both lines are pure Latin so script alone cannot sort it either.
_INK_BODY = 0.11
_INK_ROMAN = 0.42
_INK_FOOTER = 0.60
_INK_BADGE = 1.00
_INK_TOLERANCE = 0.04

_FLOW_BADGE_RE = re.compile(r"^FLOW\s+(\d+)$")
_TURN_RE = re.compile(r"^(AGENT|CALLER)\s+(.*)$", re.DOTALL)
_HEADER_FIELDS = ("INTENT", "SCENARIO", "ENTITIES", "OUTCOME")
_HEADER_RE = re.compile(rf"^({'|'.join(_HEADER_FIELDS)})\s+(.*)$", re.DOTALL)

# The agent introduces herself as "Naan <Name> pesaren" (or "...-la irunthu
# <Name> pesaren", or bare "Naan <Name>."). Read from the ROMANISED line,
# because the runtime prompt carries agent_name in Latin script -
# backend/settings.py defaults it to "Gayathri", which is flow 01's agent.
_AGENT_NAME_RE = re.compile(r"\b(?:Naan|irunthu)\s+([A-Z][a-z]+)\b")

# Outbound calls open by asking for the patient by name ("Selvi madam
# pesureengalaa?"), which is the only place in this corpus where the hospital
# demonstrably holds the patient's identity before the caller says anything.
_OUTBOUND_PATIENT_RE = re.compile(r"\b([A-Z][a-z]+)\s+(?:madam|sir|amma)\s+pesu", re.IGNORECASE)

_MRN_RE = re.compile(r"\bARV-\d{6}\b")


def expand_cids(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        cid = int(match.group(1))
        if cid not in CID_TO_TAMIL:
            raise ValueError(
                f"unmapped glyph id {cid} in {text!r} - CID_TO_TAMIL is incomplete for this PDF"
            )
        return CID_TO_TAMIL[cid]

    return _CID_RE.sub(_replace, text)


def _is_consonant(char: str) -> bool:
    return _TAMIL_CONSONANTS[0] <= char <= _TAMIL_CONSONANTS[1]


def reorder_prebase_vowels(text: str) -> str:
    out: list[str] = []
    index = 0
    end = len(text)
    while index < end:
        char = text[index]
        if char in _PREBASE_SIGNS and index + 1 < end and _is_consonant(text[index + 1]):
            consonant = text[index + 1]
            sign = char
            index += 2
            if index < end and (char, text[index]) in _SPLIT_VOWELS:
                sign = _SPLIT_VOWELS[(char, text[index])]
                index += 1
            out.append(consonant)
            out.append(sign)
        else:
            out.append(char)
            index += 1
    return "".join(out)


def validate_tamil(text: str, where: str) -> None:
    """Reject malformed clusters - the tripwire for a bad CID_TO_TAMIL entry."""
    previous = ""
    for position, char in enumerate(text):
        if char in _VOWEL_SIGNS and not _is_consonant(previous):
            context = text[max(0, position - 12) : position + 6]
            raise ValueError(f"orphan vowel sign U+{ord(char):04X} in {where}: ...{context}...")
        previous = char


def reconstruct(text: str, where: str) -> str:
    restored = reorder_prebase_vowels(expand_cids(text))
    validate_tamil(restored, where)
    return restored


def _ink_level(char: dict) -> float:
    colour = char.get("non_stroking_color")
    if isinstance(colour, (list, tuple)) and colour:
        return sum(colour) / len(colour)
    if isinstance(colour, (int, float)):
        return float(colour)
    return -1.0


def _classify(level: float) -> str:
    for name, reference in (
        ("body", _INK_BODY),
        ("roman", _INK_ROMAN),
        ("footer", _INK_FOOTER),
        ("badge", _INK_BADGE),
    ):
        if abs(level - reference) <= _INK_TOLERANCE:
            return name
    return "body"


def read_lines(pdf_path: Path) -> list[dict]:
    lines: list[dict] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for line in page.extract_text_lines(return_chars=True, strip=True):
                levels = collections.Counter(round(_ink_level(c), 3) for c in line["chars"])
                kind = _classify(levels.most_common(1)[0][0])
                if kind == "footer":
                    continue
                lines.append({"page": page_number, "text": line["text"], "kind": kind})
    return lines


def parse_flows(lines: list[dict]) -> list[dict]:
    flows: list[dict] = []
    flow: dict | None = None
    turn: dict | None = None
    field: str | None = None

    def close_turn() -> None:
        nonlocal turn
        if turn is None:
            return
        where = f"flow {flow['flow_number']} turn {len(flow['turns']) + 1}"
        flow["turns"].append(
            {
                "role": turn["role"],
                "tamil": reconstruct(" ".join(turn["tamil"]).strip(), where),
                "roman": " ".join(turn["roman"]).strip(),
            }
        )
        turn = None

    for line in lines:
        text = line["text"]
        badge = _FLOW_BADGE_RE.match(text) if line["kind"] == "badge" else None
        if badge:
            close_turn()
            flow = {
                "flow_number": int(badge.group(1)),
                "title": "",
                "intent": "",
                "scenario": "",
                "entities": [],
                "outcome": "",
                "turns": [],
            }
            flows.append(flow)
            field = "TITLE"
            continue

        if flow is None:
            continue  # cover page and contents page

        if line["kind"] == "roman":
            if turn is not None:
                turn["roman"].append(text)
            continue

        turn_start = _TURN_RE.match(text)
        if turn_start:
            close_turn()
            field = None
            turn = {"role": turn_start.group(1), "tamil": [turn_start.group(2)], "roman": []}
            continue

        if turn is not None:
            turn["tamil"].append(text)
            continue

        header = _HEADER_RE.match(text)
        if header:
            field = header.group(1)
            _append_header(flow, field, header.group(2))
            continue
        if field == "TITLE":
            flow["title"] = f"{flow['title']} {text}".strip()
            continue
        if field:
            _append_header(flow, field, text)

    close_turn()
    return flows


def _append_header(flow: dict, field: str, text: str) -> None:
    if field == "INTENT":
        flow["intent"] = f"{flow['intent']} {text}".strip()
    elif field == "SCENARIO":
        flow["scenario"] = f"{flow['scenario']} {text}".strip()
    elif field == "OUTCOME":
        flow["outcome"] = f"{flow['outcome']} {text}".strip()
    elif field == "ENTITIES":
        flow["entities"].extend(part.strip() for part in text.split(",") if part.strip())


def enrich(flow: dict) -> dict:
    """Attach the per-call facts build_dataset.py needs for {{placeholders}}.

    Only facts the agent demonstrably holds BEFORE the caller speaks belong
    here. Every MRN in this corpus is discovered mid-call - looked up from a
    mobile number, read off by the caller, or newly issued - so none of them is
    a call-start fact, and `known_mrn` stays blank. Seeding one anyway would
    train the model to speak an identifier it never looked up, and to skip the
    lookup turn that earns it: the two failure modes the runtime core's
    GROUNDING and LEDGER sections exist to prevent.
    """
    greeting = next((t for t in flow["turns"] if t["role"] == "AGENT"), None)
    greeting_roman = greeting["roman"] if greeting else ""

    agent_match = _AGENT_NAME_RE.search(greeting_roman)
    flow["agent_name"] = agent_match.group(1) if agent_match else ""

    patient_match = _OUTBOUND_PATIENT_RE.search(greeting_roman)
    flow["known_patient_name"] = patient_match.group(1) if patient_match else ""

    greeting_mrn = _MRN_RE.search(greeting["tamil"]) if greeting else None
    flow["known_mrn"] = greeting_mrn.group(0) if greeting_mrn else ""

    flow["mrns"] = list(dict.fromkeys(_MRN_RE.findall(" ".join(t["tamil"] for t in flow["turns"]))))
    return flow


# The PDF is fixed input, so these are regression pins, not defensive checks:
# each one is a detail an over-eager parser has a specific way of destroying.
def assert_known_edge_cases(flows: list[dict]) -> None:
    if len(flows) != 20:
        raise ValueError(f"expected 20 flows, parsed {len(flows)}")
    by_number = {flow["flow_number"]: flow for flow in flows}
    if sorted(by_number) != list(range(1, 21)):
        raise ValueError(f"flow numbering is not 1..20: {sorted(by_number)}")

    for flow in flows:
        if not flow["intent"]:
            raise ValueError(f"flow {flow['flow_number']} has no intent")
        if len(flow["turns"]) < 6:
            raise ValueError(f"flow {flow['flow_number']} parsed only {len(flow['turns'])} turns")
        for index, one in enumerate(flow["turns"]):
            if not one["tamil"]:
                raise ValueError(f"flow {flow['flow_number']} turn {index + 1} has empty Tamil")
        roles = [t["role"] for t in flow["turns"]]
        if roles[0] != "AGENT":
            raise ValueError(f"flow {flow['flow_number']} does not open on an AGENT turn")

    # Flow 18 is an emergency: the agent skips her own name and goes straight to
    # the address. Inventing one here would train the model to introduce itself
    # mid-emergency, which the core prompt forbids.
    emergency = by_number[18]
    if emergency["agent_name"]:
        raise ValueError(
            f"flow 18 must have no agent self-introduction, got {emergency['agent_name']!r}"
        )

    # Flow 13 registers a mother AND her newborn, so it issues two MRNs. A parser
    # that dedupes "the MRN" per flow silently loses the child's.
    registration = by_number[13]
    if registration["mrns"] != ["ARV-142771", "ARV-142772"]:
        raise ValueError(f"flow 13 must issue two MRNs, got {registration['mrns']}")

    # Flows 04 and 17 are the corpus's only outbound calls, and the only two
    # where patient_name is a call-start fact rather than something discovered.
    for number, expected in ((4, "Selvi"), (17, "Krishnamurthy")):
        found = by_number[number]["known_patient_name"]
        if found != expected:
            raise ValueError(f"flow {number} outbound patient name: expected {expected!r}, got {found!r}")
    for flow in flows:
        if flow["flow_number"] not in (4, 17) and flow["known_patient_name"]:
            raise ValueError(
                f"flow {flow['flow_number']} is inbound but claims a call-start patient name"
            )

    # Flow 20's caller genuinely trails off mid-sentence. It looks like a parsing
    # artefact and is not one - dropping it would flatten the angriest turn in
    # the corpus into a clean one.
    angry = by_number[20]
    if not any(t["role"] == "CALLER" and t["tamil"].startswith("...") for t in angry["turns"]):
        raise ValueError("flow 20 must keep the caller turn that begins with '...'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"transcript PDF not found: {args.pdf}")

    flows = [enrich(flow) for flow in parse_flows(read_lines(args.pdf))]
    assert_known_edge_cases(flows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(flows, ensure_ascii=False, indent=2), encoding="utf-8")

    turns = sum(len(flow["turns"]) for flow in flows)
    print(f"wrote {args.out} - {len(flows)} flows, {turns} turns")
    for flow in flows:
        print(
            f"  {flow['flow_number']:02d} {flow['intent']:<28} "
            f"{len(flow['turns']):>2} turns  agent={flow['agent_name'] or '-':<10} "
            f"patient={flow['known_patient_name'] or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
