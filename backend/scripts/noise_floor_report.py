"""Print how the VAD's learned noise floor moved over one call.

Tests one theory before any VAD constant is retuned (settings.py,
AudioSettings.vad_diagnostics_ms): with headphones there is no echo for the
browser's echo cancellation to remove, and automatic gain control on a close
microphone can lift the level of the pauses. _update_noise_floor() learns only
from frames the VAD calls non-speech - i.e. exactly those pauses - so an
AGC-lifted pause would raise the learned floor over the call, raise the onset
bar with it (floor x VAD_ONSET_SNR), and make a turn progressively harder to
start.

What confirms it:   the floor in quiet stretches rises from the first third of
                    the call to the last, the onset bar rises past the
                    caller's normal speech level, and onset_refusals (hops
                    the VAD flagged as speech that the loudness gate turned
                    away) climb with it.
What rules it out:  a flat floor across the call, and refusals that do not
                    grow.

Record a call with the server started as VAD_DIAGNOSTICS_MS=500, speak
normally with pauses for a couple of minutes, hang up, then:

    python -m backend.scripts.noise_floor_report            # latest call
    python -m backend.scripts.noise_floor_report <connection_id>
    python -m backend.scripts.noise_floor_report --every 2  # one row per 2 s
"""

from __future__ import annotations

import argparse
import statistics

from backend.persistence import CallEventStore
from backend.settings import PersistenceSettings


def _quiet(sample: dict) -> bool:
    """A stretch the floor is actually learned from: no turn open, agent silent."""
    return not sample["in_speech"] and not sample["agent_audible"]


def _thirds(samples: list[dict]) -> list[list[dict]]:
    size = max(1, len(samples) // 3)
    return [samples[:size], samples[size : 2 * size], samples[2 * size :]]


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None) -> str:
    return "   -" if value is None else f"{value:7.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("connection_id", nargs="?", help="call to report on (default: the most recent)")
    parser.add_argument("--every", type=float, default=0.0, help="print at most one row per this many seconds")
    args = parser.parse_args()

    store = CallEventStore(PersistenceSettings())
    store.load()
    connection_id = args.connection_id
    if connection_id is None:
        calls = store.recent_calls(limit=1)
        if not calls:
            raise SystemExit("no calls in the call log yet")
        connection_id = calls[0]["connection_id"]

    events = store.events_for_call(connection_id)
    samples = [e for e in events if e.get("type") == "vad_diagnostics"]
    mic = next((e.get("settings") for e in events if e.get("type") == "mic_settings"), None)
    turns = sum(1 for e in events if e.get("type") == "vad_start")

    print(f"call {connection_id}")
    print(f"mic as applied by the browser: {mic or 'not reported (older client)'}")
    if not samples:
        raise SystemExit("no vad_diagnostics in this call - start the server with VAD_DIAGNOSTICS_MS=500")

    print(f"{len(samples)} samples over {samples[-1]['t']:.0f} s, {turns} turns opened\n")
    print("     t   floor   bar     room   p_max  state        refused")
    last_printed = -1e9
    for s in samples:
        if s["t"] - last_printed < args.every:
            continue
        last_printed = s["t"]
        state = "TURN" if s["in_speech"] else ("agent" if s["agent_audible"] else "quiet")
        print(
            f"{s['t']:6.1f} {s['noise_floor']:7.1f} {s['onset_bar']:7.1f} {s['room_rms']:7.1f}"
            f"  {s['p_max']:.2f}   {state:<10} {s['onset_refusals']:5d}"
        )

    print("\nBy third of the call (medians; floor and quiet-room over QUIET samples only):")
    print("  third   floor    bar     quiet-room  speech-room  refusals")
    for index, part in enumerate(_thirds(samples), start=1):
        quiet = [s for s in part if _quiet(s)]
        talking = [s for s in part if s["in_speech"]]
        print(
            f"  {index}     {_fmt(_median([s['noise_floor'] for s in quiet]))}"
            f" {_fmt(_median([s['onset_bar'] for s in part]))}"
            f"     {_fmt(_median([s['room_rms'] for s in quiet]))}"
            f"      {_fmt(_median([s['room_rms'] for s in talking]))}"
            f"     {sum(s['onset_refusals'] for s in part):5d}"
        )

    first, last = _thirds(samples)[0], _thirds(samples)[-1]
    floor_start = _median([s["noise_floor"] for s in first if _quiet(s)])
    floor_end = _median([s["noise_floor"] for s in last if _quiet(s)])
    speech = _median([s["room_rms"] for s in samples if s["in_speech"]])
    if floor_start and floor_end:
        print(f"\nfloor drift, first third -> last third: x{floor_end / floor_start:.2f}")
    if speech:
        bar_end = _median([s["onset_bar"] for s in last])
        print(f"caller speech level {speech:.0f} vs onset bar at the end {bar_end:.0f}"
              f" (x{speech / bar_end:.2f}; a turn opens only on hops louder than the bar)")


if __name__ == "__main__":
    main()
