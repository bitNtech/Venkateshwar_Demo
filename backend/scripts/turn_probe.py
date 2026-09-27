"""Time a real agent turn end to end: when text arrives vs when audio arrives.

Answers the only question that matters: is the silence the caller hears the
LLM thinking, or TTS synthesizing?
"""
import asyncio, json, time, sys
import websockets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TURN = sys.argv[1] if len(sys.argv) > 1 else "Cardiology-la oru appointment book pannanum."


async def main():
    async with websockets.connect("ws://127.0.0.1:8000/ws/audio", max_size=None) as ws:
        await ws.send(json.dumps({
            "type": "call_started", "audio_format": "pcm_s16le",
            "sample_rate": 16000, "channels": 1, "language": "ta",
        }))

        # Let the scripted greeting finish so it isn't mixed into the timings.
        t_start = time.perf_counter()
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            if isinstance(msg, str) and json.loads(msg).get("type") == "agent_speaking_end":
                break
        print(f"[greeting done at {time.perf_counter() - t_start:.2f}s]\n")

        await ws.send(json.dumps({"type": "user_text", "text": TURN}))
        t0 = time.perf_counter()
        print(f"CALLER: {TURN}\n")

        audio_total = 0.0
        last_audio_end = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=180)
            t = time.perf_counter() - t0
            if isinstance(msg, bytes):
                secs = len(msg) / 2 / 24000
                audio_total += secs
                # A gap = audio arriving after the previously delivered audio
                # would have finished playing. That is the silence heard.
                gap = "" if last_audio_end is None else f"   GAP {max(0.0, t - last_audio_end):5.2f}s"
                print(f"{t:6.2f}s  AUDIO  {secs:4.2f}s of speech{gap}")
                last_audio_end = max(t, last_audio_end or 0) + secs
                continue
            e = json.loads(msg)
            kind = e.get("type")
            if kind == "agent_clause":
                print(f"{t:6.2f}s  TEXT   {e['text'][:60]}")
            elif kind == "agent_tool_call":
                print(f"{t:6.2f}s  TOOL   {e['name']}")
            elif kind in ("agent_speaking_start",):
                print(f"{t:6.2f}s  speaking_start")
            elif kind == "agent_speaking_end":
                print(f"{t:6.2f}s  speaking_end")
                break
            elif kind in ("agent_error", "agent_audio_error"):
                print(f"{t:6.2f}s  ERROR  {e.get('message')}")

        print(f"\ntotal speech {audio_total:.2f}s, wall clock {time.perf_counter() - t0:.2f}s")
        await ws.send(json.dumps({"type": "call_ended"}))


asyncio.run(main())
