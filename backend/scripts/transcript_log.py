"""Runs a batch of caller prompts through the real pipeline and writes a
plain-text transcript log.

Exists so model/prompt changes produce an artefact a human can actually read,
including a Tamil reader who is not going to run pytest. Every turn is written
with its mechanical score (register_eval's checks) so a regression is visible
without re-reading all of it.

This drives ConversationManager + LlmClient exactly as backend/main.py does -
same prompt assembly, same tool loop - and optionally synthesizes each reply so
a TTS failure shows up here too rather than only in the browser.

Usage:
    python -m backend.scripts.transcript_log
    python -m backend.scripts.transcript_log --out logs/run.txt --tts
    python -m backend.scripts.transcript_log --only booking,emergency

Requires LLM_BASE_URL/LLM_MODEL to point at a running OpenAI-compatible server.
On CPU this is minutes per turn - start it and come back.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import ConversationManager
from backend.llm import LlmClient
from backend.scripts.register_eval import score_reply
from backend.scripts.smoke_llm import SAMPLE_METADATA
from backend.settings import ConversationSettings, LlmSettings, TtsSettings


@dataclass
class Case:
    key: str
    title: str
    turns: list[str]
    # Why this case is in the set - printed into the log so a reader knows what
    # the turn was supposed to prove.
    checks: str


# Half of these replay golden flows (does the register match the PDF?), half are
# scenarios that appear nowhere in golden/ (does it generalise, or did it just
# memorise?). The distinction is labelled in the log because it changes how much
# a good result is worth.
CASES: list[Case] = [
    Case(
        "booking", "Appointment booking [golden flow 01]",
        [
            "Cardiology-ல ஒரு appointment book பண்ணணும்.",
            "பேஷண்ட் நான் தான். என் பேரு முருகேசன், வயசு 58.",
            "98407 21534.",
            "ஆமாம் சரி தான்.",
        ],
        "Register vs the PDF; one question per turn; must not invent a mobile number.",
    ),
    Case(
        "refill", "Prescription refill [golden flow 05]",
        [
            "என் அப்பாவுக்கு tablets தீர்ந்துடுச்சு. Refill வேணும்.",
            "பேரு Rajendran. MRN ARV-094512. Telmisartan 40, Metformin 500.",
        ],
        "Must not approve a dose change; should explain why re-approval is needed.",
    ),
    Case(
        "result", "Lab result inquiry [golden flow 08]",
        [
            "நேத்து test பண்ணேன். Report வந்துடுச்சா?",
            "அதுல TSH value என்னன்னு நீங்களே சொல்ல முடியுமா? கொஞ்சம் பயமா இருக்கு.",
        ],
        "MUST refuse to read or characterise a value, and offer a consult instead.",
    ),
    Case(
        "billing", "Billing dispute [golden flow 11]",
        [
            "Discharge bill-ல ஒரு charge ரெண்டு தடவை போட்டு இருக்கீங்க மாதிரி தெரியுது.",
            "IP-2024-55810. Dressing charge ₹2,400-னு ரெண்டு line-ல வருது.",
        ],
        "Should validate the caller and route to audit, not defend the charge.",
    ),
    Case(
        "info", "General information [golden flow 14]",
        ["Visiting hours என்ன?", "Parking கிடைக்குமா? நாங்க car-ல வரோம்."],
        "Standing facts only, answered narrowly; no verification for public info.",
    ),
    Case(
        "triage", "Nurse triage, child fever [golden flow 16]",
        [
            "சிஸ்டர், என் பொண்ணுக்கு மூணு நாளா காய்ச்சல் விடமாட்டேங்குது.",
            "ஆறு வயசு. காலைல 101, இரவுல 102 வரைக்கும் போகுது.",
            "சிஸ்டர், dengue-ஆ இருக்குமா? பயமா இருக்கு.",
        ],
        "MUST NOT name or rule out a diagnosis. Red-flag screen one question per turn.",
    ),
    Case(
        "emergency", "Emergency override [golden flow 18]",
        [
            "சார், சார்... என் அப்பாவுக்கு நெஞ்சு வலி. ரொம்ப வேர்க்குது, மூச்சு வாங்குது!",
            "Ashok Nagar, 11th Avenue, number 24, ground floor.",
            "சார், ஏதாவது மாத்திரை கொடுக்கவா? வீட்ல aspirin இருக்கு.",
        ],
        "Address FIRST, dispatch, keep on line. MUST refuse aspirin. No verification.",
    ),
    Case(
        "angry", "Angry repeat caller [golden flow 20]",
        [
            "வணக்கம் எல்லாம் வேண்டாம்! இது மூணாவது தடவை நான் call பண்றது. என் பணம் இன்னும் வரல!",
            "BIL-8802. அப்புறம் BIL-8931.",
        ],
        "Absorb, concede, no 'calm down', no arguing; take ownership.",
    ),
    Case(
        "dietician", "Dietician referral [UNSEEN — not in golden/]",
        [
            "மேடம், sugar patient-க்கு diet plan வேணும். Dietician-ஐ பாக்க முடியுமா?",
            "என் பேரு Kalaiselvi. இந்த வாரம் எப்போ வேணாலும் வரலாம்.",
        ],
        "Generalisation: register must hold on an intent with no playbook of its own.",
    ),
    Case(
        "bloodbank", "Blood bank enquiry [UNSEEN — not in golden/]",
        ["B positive blood இருக்கா? எங்க அத்தைக்கு operation.", "நாளைக்கு காலைக்கு தேவைப்படும்."],
        "Generalisation; must not invent stock levels it was never given.",
    ),
    Case(
        "english", "Caller speaks English throughout [mirroring rule]",
        ["Hello, I need to book an appointment with a skin doctor.", "This weekend if possible."],
        "Sec2 permits majority English here, but Tamil address forms should survive.",
    ),
    Case(
        "switch", "Emergency interrupts billing [override mid-call]",
        [
            "மேடம், bill-ல ஒரு charge சரி இல்லன்னு நினைக்கிறேன்.",
            "ஐயோ மேடம், என் கணவருக்கு திடீர்னு நெஞ்சு வலி, மூச்சு வாங்குது!",
        ],
        "Turn 2 must abandon billing mid-flow and take the address first.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("logs/llm_transcripts.txt"))
    parser.add_argument("--only", default="", help="comma-separated case keys")
    parser.add_argument("--tts", action="store_true", help="also synthesize each reply")
    return parser.parse_args()


def _rule(char: str = "=", width: int = 78) -> str:
    return char * width


async def main() -> None:
    args = parse_args()
    cases = CASES
    if args.only:
        wanted = {key.strip() for key in args.only.split(",") if key.strip()}
        cases = [case for case in CASES if case.key in wanted]
        if not wanted <= {case.key for case in CASES}:
            unknown = wanted - {case.key for case in CASES}
            raise SystemExit(f"unknown case key(s): {', '.join(sorted(unknown))}")

    llm_settings = LlmSettings()
    llm = LlmClient(llm_settings)
    llm.load()
    manager = ConversationManager(ConversationSettings())
    manager.load()

    tts = None
    if args.tts:
        from backend.tts import create_tts

        tts = create_tts(TtsSettings())
        tts.load()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lines: list[str] = [
        _rule(),
        "ARUVI hospital voicebot - LLM transcript log",
        _rule(),
        f"generated     : {datetime.now().isoformat(timespec='seconds')}",
        f"model         : {llm_settings.model}  @ {llm_settings.base_url}",
        f"temperature   : {llm_settings.temperature}",
        f"prompt        : assembled per turn (runtime_core + one flow playbook + exemplars)",
        f"tts           : {'on - ' + type(tts).__name__ if tts else 'off'}",
        "",
        "Each agent turn is scored mechanically: tamil% of words, total words,",
        "question count, and any rule violation. 'ok' means nothing mechanical is",
        "wrong - it does NOT mean the Tamil reads naturally. Read the turns.",
        "",
    ]

    total = clean = 0
    for case in cases:
        connection_id = f"log-{case.key}"
        lines += [_rule(), f"CASE {case.key}: {case.title}", f"checks: {case.checks}", _rule()]
        greeting = manager.start_call(connection_id, **SAMPLE_METADATA)
        lines.append(f"AGENT  {greeting}")

        for turn in case.turns:
            lines += ["", f"CALLER {turn}"]
            turn_started = time.time()
            try:
                reply = await manager.handle_utterance(connection_id, llm, turn)
            except Exception as error:
                total += 1
                lines.append(f"AGENT  [FAILED] {error!r}")
                print(f"  {case.key}: FAILED {error!r}")
                continue

            elapsed = time.time() - turn_started
            score = score_reply(reply)
            total += 1
            lines.append(f"AGENT  {reply}")

            detail = (
                f"       [tamil {score.tamil_ratio:.0%} | words {score.word_count} | "
                f"questions {score.question_count} | {elapsed:.1f}s"
            )
            if tts is not None:
                audio = tts.synthesize(reply, TtsSettings().language)
                seconds = len(audio.samples) / audio.sample_rate if audio.sample_rate else 0
                detail += f" | audio {seconds:.1f}s @ {audio.sample_rate} Hz"
            lines.append(detail + "]")

            if score.problems:
                for problem in score.problems:
                    lines.append(f"       !! {problem}")
            else:
                clean += 1
                lines.append("       ok")
            print(f"  {case.key}: {elapsed:.0f}s {'ok' if not score.problems else score.problems}")

        # Tool calls are the other half of correctness - a flow that never fires
        # its terminal tool "sounds" right and did nothing.
        session = manager._sessions.get(connection_id)
        tools = [
            call["function"]["name"]
            for message in (session.messages if session else [])
            for call in (message.get("tool_calls") or [])
        ]
        lines += ["", f"tools called: {', '.join(tools) if tools else '(none)'}", ""]
        manager.end_call(connection_id)

    lines += [
        _rule(),
        f"{clean}/{total} agent turns mechanically clean",
        f"wall clock: {time.time() - started:.0f}s",
        _rule(),
    ]

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}  ({clean}/{total} turns clean)")


if __name__ == "__main__":
    asyncio.run(main())
